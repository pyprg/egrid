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

def get_terminals(model, additional_terminals):
    """Creates a pandas.DataFrame of terminals.

    Parameters
    ----------
    model: model.Model

    additional_terminals: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        * .id_of_node
        * .id_of_device
        * .devtype 'branch'|'bridge'|'injection'"""
    cols_br = ['id_of_node', 'id_of_branch']
    cols_inj = ['id_of_node', 'id']
    branchterms = model.branchterminals
    bra_terms = (
        branchterms[cols_br].rename(columns={'id_of_branch':'id_of_device'}))
    bra_terms['devtype'] = 'branch'
    bridgeterms = model.bridgeterminals
    bri_terms = (
        bridgeterms[cols_br].rename(columns={'id_of_branch':'id_of_device'}))
    bri_terms['devtype'] = 'bridge'
    inj_terms = (
        model.injections[cols_inj].rename(columns={'id':'id_of_device'}))
    inj_terms['devtype'] ='injection'
    terms = [bra_terms, bri_terms, inj_terms]
    if additional_terminals is not None:
        terms.append(additional_terminals.assign(devtype='batch'))
    return pd.concat(terms, axis=0)

def get_node_device_graph(model):
    """Creates a bipartite directed graph of an electrical model.

    Connectivity nodes and devices are vertices of the graph. Terminals
    are the edges. The first set of vertices are the connectivity nodes.
    The other set of vertices contains devices.

    Parameters
    ----------
    model: model.Model

    Returns
    -------
    networkx.DiGraph"""
    edges = get_terminals(model, None)
    return nx.from_pandas_edgelist(
        edges,
        source='id_of_node',
        target='id_of_device',
        create_using=nx.DiGraph)

def get_outputs(model):
    """

    Parameters
    ----------
    model: egrid.model.Model

    Returns
    -------
    pandas.DataFrame
        * .id_of_batch, str
        * .id_of_node, str
        * .id_of_device, str"""
    branchoutputs = (
        model.branchoutputs[['id_of_batch', 'id_of_branch', 'id_of_node']]
        .rename(columns={'id_of_branch':'id_of_device'}))
    injectionoutputs = (
        model.injectionoutputs
        .join(model.injections, on='index_of_injection')
        [['id_of_batch', 'id_of_injection', 'id_of_node']]
        .rename(columns={'id_of_injection':'id_of_device'}))
    return pd.concat([branchoutputs, injectionoutputs])

def get_batches_with_type(model, outputs):
    """Returns information on P/Q/I-measurements of batches.

    Parameters
    ----------
    model: model.Model

    outputs: pandas.DataFrame
        * .id_of_batch

    Returns
    -------
    pandas.DataFrame (id_of_batch)
        * .P, bool
        * .Q, bool
        * .I, bool"""
    id_of_batch = pd.Series(outputs.id_of_batch.unique())
    return pd.DataFrame(
        {'id_of_batch': id_of_batch,
         'P': id_of_batch.isin(model.pvalues.id_of_batch),
         'Q': id_of_batch.isin(model.qvalues.id_of_batch),
         'I': id_of_batch.isin(model.ivalues.id_of_batch)})

def split(digraph, *, nodes=(), terminals=()):
    """Removes terminals and nodes from digraph. Returns subgraphs.

    Filtered out nodes also filter out any of their edges.

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
    terminals with specified flow measurements.

    Parameters
    ----------
    model: egrid.model.Model

    Returns
    -------
    function
        (list of 'P', 'Q', 'I') -> (iterator over networkx.Graph)"""
    outputs = get_outputs(model)
    batches_with_type = get_batches_with_type(model, outputs)
    # enhancement of graph, new nodes are given the ids of the batch
    additional_edges = (
        outputs[['id_of_batch', 'id_of_device']]
        .rename(columns={'id_of_batch':'id_of_node'}))
    edges = get_terminals(model, additional_edges)
    digraph = nx.from_pandas_edgelist(
        edges,
        source='id_of_node',
        target='id_of_device',
        create_using=nx.DiGraph)
    def make_subgraphs(output_types):
        """Returns splitted graph.

        Creates a directed graph splitted at terminals having
        at least one output of given output type. Subgraphs contain
        additional nodes and edges, which are not part of the model,
        at locations where the original graph was splitted.

        There are two sets of nodes. First set contains connectivity nodes.
        The second set contains the devices.

        Parameters
        ----------
        output_types: list
            'P', 'Q', 'I'

        Returns
        -------
        iterable
            networkx.Graph"""
        assert isinstance(output_types, list), \
            "argument output_types shall be an instance of list"
        assert all(x in 'PQI' for x in output_types), (
            "each element must one of 'P', 'Q' or 'I' "
            f"but is {output_types}")
        # id_of_batch[str], P[bool], Q[bool], I[bool]
        selected_batch = batches_with_type[output_types].apply(any, axis=1)
        # additional nodes have ID of batch, hide them if not selected
        nodes_to_hide = (
            batches_with_type.id_of_batch[~selected_batch].to_list())
        to_hide = outputs.id_of_batch.isin(
            batches_with_type.id_of_batch[selected_batch])
        terminals_to_hide = (
            outputs.loc[to_hide, ['id_of_node', 'id_of_device']]
            .apply(tuple, axis=1)
            .to_list())
        # function 'networkx.restricted_view' filters out edges connected
        #   to filtered out nodes, hence, there is no need to pass
        #   corresponding edges to function 'split'
        return split(digraph, nodes=nodes_to_hide, terminals=terminals_to_hide)
    return make_subgraphs

# def _get_group_of_injections(idx_of_group, digraph):
#     cns, devices = bipartite.sets(digraph)
#     connodes = digraph.subgraph(cns).nodes(data='is_slack')
#     devnodes = digraph.subgraph(devices).nodes(data='is_injection')
#     df = pd.DataFrame(
#         {'id_of_injection': (dn[0] for dn in devnodes if dn[1]),
#          'index_of_group': idx_of_group})
#     return df, any(cn[1] for cn in connodes)

# def get_injection_groups(subgraphs):
#     """Collects injections subgraph.

#     Parameters
#     ----------
#     subgraphs: networkx.Digraph
#         the graph is bipartite, terminals of devices are the edges, the
#         graph has two sets of vertices

#         * connectivity nodes
#         * devices (electric branches and injections)

#     Returns
#     -------
#     tuple
#         pandas.DataFrame
#             * .index_of_group, int
#             * .has_injection, bool
#             * .has_slack, bool

#         pandas.DataFrame
#             * .id_of_injection, str
#             * .index_of_group, int"""
#     dfs = []
#     group_info = []
#     for idx_of_group, digraph in enumerate(subgraphs):
#         df, has_slack = _get_group_of_injections(idx_of_group, digraph)
#         if not df.empty:
#             dfs.append(df)
#         group_info.append((idx_of_group, not df.empty, has_slack))
#     df_injections = (
#         pd.concat(dfs) if dfs else
#         pd.DataFrame([], columns=['id_of_injection', 'index_of_group']))
#     df_group_info = pd.DataFrame(
#         group_info, columns=['index_of_group', 'has_injection', 'has_slack'])
#     df_group_info.sort_values('index_of_group', inplace=True)
#     return df_group_info, df_injections

# def get_nodes_and_injections(digraph):
#     return nx.to_pandas_edgelist(
#         digraph, source='id_of_node', target='id_of_device')

# _pq_PQ = ('p', ['P']), ('q', ['Q'])
# _pq_PQI = ('p', ['P', 'I']), ('q', ['Q', 'I'])

# def get_injection_group_info(model, with_I_values=True):
#     """Returns two collections of injection groups one for active power
#     scaling and one for reactive power scaling.

#     Parameters
#     ----------
#     model: egrid.model.Model

#     with_I_values: bool
#         optional, default True
#         if true split graph at terminals with IValues (or P or Q)
#         else only at terminals with (P or Q)

#     Returns
#     ------
#     dict
#         * ['p'], tuple
#             * pandas.DataFrame
#                 * .index_of_group, int
#                 * .has_injection, bool
#                 * .has_slack, bool
#             * pandas.DataFrame
#                 * .id_of_injection, str
#                 * .index_of_group, int
#         * ['q'], tuple
#             * pandas.DataFrame
#                 * .index_of_group, int
#                 * .has_injection, bool
#                 * .has_slack, bool
#             * pandas.DataFrame
#                 * .id_of_injection, str
#                 * .index_of_group, int"""
#     make_subgraphs = get_make_subgraphs(model)
#     return {
#         part: make_subgraphs(types_of_values)
#         for part, types_of_values in (_pq_PQI if with_I_values else _pq_PQ)}



