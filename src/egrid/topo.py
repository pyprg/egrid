# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 pyprg

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Created on Fri Aug 25 20:26:34 2023

@author: pyprg
"""

import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from networkx.classes.filters import hide_diedges, hide_nodes

def get_node_device_graph(model, additional_edges=None):
    """Creates a bipartite, directed graph of devices and connectivity nodes.

    Branches, injections and connectivity nodes of the electric grid
    abstraction are the vertices, terminals of the grid are the edges of the
    new created graph.

    The created graph is bipartite in that connectivity nodes is one set of
    vertices and branches/injections build the other set of vertices.
    Connectivity nodes are connected via edges (terminals) to
    branches/injections only and branches/injections only to connectivity
    nodes. Connectivity nodes have outgoing edges and branches/injections
    have incoming edges only.

    Parameters
    ----------
    model: model.Model

    additional_edges: pandas.DataFrame
        optional, default is None

        * .id_of_node, str
        * .id_of_device, str

    Returns
    -------
    networkx.DiGraph
        vertices have a bool attributes 'is_injection' and 'is_slack'

        * first set is made of connectivity nodes
        * second set is made of branches/bridges/injections"""
    cols_br = ['id_of_node', 'id_of_branch']
    cols_inj = ['id_of_node', 'id']
    branchterms = model.branchterminals
    bra_terms = (
        branchterms[cols_br].rename(columns={'id_of_branch':'id_of_device'}))
    bridgeterms = model.bridgeterminals
    bri_terms = (
        bridgeterms[cols_br].rename(columns={'id_of_branch':'id_of_device'}))
    inj_terms = (
        model.injections[cols_inj].rename(columns={'id':'id_of_device'}))
    terms = [bra_terms, bri_terms, inj_terms]
    if additional_edges is not None:
        terms.append(additional_edges)
    edges = pd.concat(terms, axis=0)
    digraph =  nx.from_pandas_edgelist(
        edges,
        source='id_of_node',
        target='id_of_device',
        create_using=nx.DiGraph)
    nx.set_node_attributes(digraph, False, 'is_injection')
    is_injection = pd.Series(True, index=inj_terms.id_of_device)
    nx.set_node_attributes(digraph, is_injection, 'is_injection')
    nx.set_node_attributes(digraph, False, 'is_slack')
    is_slack = pd.Series(True, index=model.slacks.id_of_node)
    nx.set_node_attributes(digraph, is_slack, 'is_slack')
    return digraph

def split(digraph, *, nodes=(), terminals=()):
    """Removes terminals and nodes from digraph. Returns subgraphs.

    Parameters
    ----------
    digraph: networkx.DiGraph
        digraph is bipartite, one set of vertices are the connectivity nodes,
        the other set of vertices are branches, edges are outgoing from 
        connectivity nodes and incoming at branches

    terminals: iterable
        optional,
        tuple

        * .id_of_node, str
        * .id_of_branch, str

    nodes: iterable
        optional,
        str, id of node

    Returns
    -------
    iterator
        networkx.DiGraph"""
    view = nx.restricted_view(digraph, nodes=nodes, edges=terminals)
    return (view.subgraph(c) for c in nx.weakly_connected_components(view))

def get_make_subgraphs(model):
    """Creates a function which returns graphs of the grid splitted at
    terminals with flow specified measurements.

    Parameters
    ----------
    model: egrid.model.Model

    Returns
    -------
    function
        (list of 'P', 'Q', 'I') -> (iterator over networkx.Graph)"""
    branchoutputs = (
        model.branchoutputs[['id_of_batch', 'id_of_branch', 'id_of_node']]
        .rename(columns={'id_of_branch':'id_of_device'}))
    injectionoutputs = (
        model.injectionoutputs
        .join(model.injections, on='index_of_injection')
        [['id_of_batch', 'id_of_injection', 'id_of_node']]
        .rename(columns={'id_of_injection':'id_of_device'}))
    outputs = pd.concat([branchoutputs, injectionoutputs])
    # batches with more than one output only
    output_count = outputs.id_of_batch.value_counts()
    selected_batches = output_count.index[1 < output_count]
    selected_outputs = outputs[outputs.id_of_batch.isin(selected_batches)]
    # enhancement of graph
    additional_edges = (
        selected_outputs[['id_of_batch', 'id_of_device']]
        .rename(columns={'id_of_batch':'id_of_node'}))
    # prepare splitting at terminals with flow measurements
    #   if multiple outputs (having the same ID) are connected to one 
    #   connectivity node it needs to be split for tracing in one node 
    #   connecting the concerning terminals and one node connecting the 
    #   rest of the terminals, this ensures that pathes starting
    #   at those terminals are part of the correct group
    extra_edges = selected_outputs.apply(
        lambda row: (row.id_of_batch, row.id_of_device), axis=1)
    extra_edges.name = "extra_edge"
    # original
    batch_edges = selected_outputs.apply(
        lambda row: (row.id_of_node, row.id_of_device), axis=1)
    batch_edges.name = "edge_of_batch"
    edges = pd.concat([
        extra_edges, batch_edges, selected_outputs.id_of_batch],
        axis=1)
    #
    batches_with_type = pd.DataFrame(
        {'id_of_batch': selected_batches,
         'P': selected_batches.isin(model.pvalues.id_of_batch),
         'Q': selected_batches.isin(model.qvalues.id_of_batch),
         'I': selected_batches.isin(model.ivalues.id_of_batch)})
    # create the digraph
    digraph = get_node_device_graph(model, additional_edges)
    def make_subgraphs(output_types):
        """Returns splitted graph.

        Creates a directed graph splitted at terminals having
        at least one output of given output type. The graph can contain
        additional nodes which are not part of the model.

        There are two sets of nodes. First set contains connectivity nodes.
        The second set contains the devices. Devices have a bool attribute
        'is_injection'.

        Parameters
        ----------
        output_types: list
            'P', 'Q', 'I'

        Returns
        -------
        iterable
            networkx.Graph"""
        assert  isinstance(output_types, list), \
            "argument output_types shall be an instance of list"
        assert all(x in ['P', 'Q', 'I'] for x in output_types), (
            "each element must one of 'P', 'Q' or 'I' "
            f"but were {output_types}")
        apply_extra = batches_with_type[output_types].apply(any, axis=1)
        nodes_to_hide = batches_with_type.id_of_batch[~apply_extra].to_list()
        extra = edges.id_of_batch.isin(
            batches_with_type[apply_extra].id_of_batch)
        edges_to_hide = (
            pd.concat([edges.extra_edge[~extra], edges.edge_of_batch[extra]])
            .to_list())
        return split(digraph, nodes=nodes_to_hide, terminals=edges_to_hide)
    return make_subgraphs

def _get_group_of_injections(idx_of_group, digraph):
    cns, devices = bipartite.sets(digraph)
    connodes = digraph.subgraph(cns).nodes(data='is_slack')
    devnodes = digraph.subgraph(devices).nodes(data='is_injection')
    df = pd.DataFrame(
        {'id_of_injection': (n[0] for n in devnodes if n[1]),
         'idx_of_group': idx_of_group})
    return df, any(cn[1] for cn in connodes)

def get_injection_groups(subgraphs):
    dfs = []
    group_info = []
    for idx_of_group, digraph in enumerate(subgraphs):
        df, has_slack = _get_group_of_injections(idx_of_group, digraph)
        if not df.empty:
            dfs.append(df)
        group_info.append((idx_of_group, not df.empty, has_slack))
    df_injections = (
        pd.concat(dfs) if dfs else
        pd.DataFrame([], columns=['id_of_injecion', 'idx_of_group']))
    df_group_info = pd.DataFrame(
        group_info, columns=['idx_of_group', 'has_injection', 'has_slack'])
    return df_group_info, df_injections

