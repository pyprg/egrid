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

    Resulting data include branch terminals, bridge terminals and
    terminals of injections.

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
        branchterms[cols_br].rename(columns={'id_of_branch': 'id_of_device'}))
    bra_terms['devtype'] = 'branch'
    bridgeterms = model.bridgeterminals
    bri_terms = (
        bridgeterms[cols_br].rename(columns={'id_of_branch': 'id_of_device'}))
    bri_terms['devtype'] = 'bridge'
    inj_terms = (
        model.injections[cols_inj].rename(columns={'id': 'id_of_device'}))
    inj_terms['devtype'] = 'injection'
    terms = [bra_terms, bri_terms, inj_terms]
    if additional_terminals is not None:
        terms.append(additional_terminals.assign(devtype='batch'))
    return pd.concat(terms, axis=0)

def get_node_device_graph(model):
    """Creates a bipartite directed graph of an electrical model.

    Connectivity nodes and devices are vertices of the graph. Terminals
    are the edges. The first set of vertices are the connectivity nodes.
    The other set of vertices contains devices.

    The function is made for testing.

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
    """Concatenates and renames columns of injection- and branch-outputs.

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
        .rename(columns={'id_of_branch': 'id_of_device'}))
    injectionoutputs = (
        model.injectionoutputs
        .join(model.injections, on='index_of_injection')
        [['id_of_batch', 'id_of_injection', 'id_of_node']]
        .rename(columns={'id_of_injection': 'id_of_device'}))
    if branchoutputs.empty:
        return injectionoutputs
    if injectionoutputs.empty:
        return branchoutputs
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

# def split(digraph, *, nodes=(), terminals=()):
#     """Removes terminals and nodes from digraph. Returns subgraphs.

#     Filtered out nodes also filter out any of their edges.

#     Parameters
#     ----------
#     digraph: networkx.DiGraph
#         digraph is bipartite, one set of vertices are the connectivity nodes,
#         the other set of vertices are branches, edges are outgoing from
#         connectivity nodes and incoming at branches

#     terminals: iterable
#         optional,
#         tuple

#         * .id_of_node, str
#         * .id_of_branch, str

#     nodes: iterable
#         optional,
#         str, id of node

#     Returns
#     -------
#     iterator
#         networkx.DiGraph"""
#     view = nx.restricted_view(digraph, nodes=nodes, edges=terminals)
#     return (
#         view.subgraph(c)
#         for c in nx.weakly_connected_components(view))

def split(digraph, edges, *, nodes=(), terminals=()):
    """Removes terminals and nodes from digraph. Returns subgraphs.

    Filtered out nodes also filter out any of their edges.

    Parameters
    ----------
    digraph: networkx.DiGraph
        digraph is bipartite, one set of vertices are the connectivity nodes,
        the other set of vertices are branches, edges are outgoing from
        connectivity nodes and incoming at branches

    edges: pandas.DataFrame
        * .id_of_node, str
        * .id_of_device, str

    terminals: iterable
        optional, edges to be removed from digraph
        tuple

        * .id_of_node, str
        * .id_of_branch, str

    nodes: iterable
        optional, nodes to be removed from digraph
        str, id of node

    Returns
    -------
    iterator
        tuple, subgraph data
            * networkx.OutEdgeView
            * pandas.Series, IDs of connectivity nodes
            * pandas.Series, IDs of devices"""
    # bipartite.sets fails for one connectivity-node graphs,
    #   therefore, implemented this way
    view = nx.restricted_view(digraph, nodes=(), edges=terminals)
    for wcc in nx.weakly_connected_components(view):
        sub = digraph.subgraph(wcc)
        nodes = set(sub.nodes)
        is_edge = edges.id_of_node.isin(nodes)
        is_device = edges.id_of_device.isin(nodes)
        yield (
            sub.edges,
            set(edges.id_of_node[is_edge]),
            set(edges.id_of_device[is_device]))

def get_make_subgraphs(model, outputs, batches_with_type):
    """Creates a function which returns graphs of the grid splitted at
    terminals with specified flow measurements.

    Parameters
    ----------
    model: egrid.model.Model

    outputs: pandas.DataFrame
        * .id_of_batch, str
        * .id_of_node, str
        * .id_of_device, str
    batches_with_type: pandas.DataFrame (id_of_batch)
        * .P, bool
        * .Q, bool
        * .I, bool"

    Returns
    -------
    function
        (list of 'P', 'Q', 'I') -> (iterator over networkx.Graph)"""
    # enhancement of graph, new nodes are given the ids of the batch
    additional_edges = (
        outputs[['id_of_batch', 'id_of_device']]
        .rename(columns={'id_of_batch': 'id_of_node'}))
    all_edges = get_terminals(model, additional_edges)
    digraph = nx.from_pandas_edgelist(
        all_edges,
        source='id_of_node',
        target='id_of_device',
        create_using=nx.DiGraph)
    def make_subgraphs(flow_types):
        """Returns splitted graph.

        Creates a directed graph splitted at terminals having
        at least one output of given output type. Subgraphs contain
        additional nodes and edges, which are not part of the model,
        at locations where the original graph was splitted.

        There are two sets of nodes. First set contains connectivity nodes.
        The second set contains the devices.

        Parameters
        ----------
        flow_types: list
            'P', 'Q', 'I'

        Returns
        -------
        iterator
            tuple, subgraph data
                * networkx.OutEdgeView
                * pandas.Series, IDs of connectivity nodes
                * pandas.Series, IDs of devices"""
        assert isinstance(flow_types, list), \
            "argument flow_types shall be an instance of list"
        assert all(x in 'PQI' for x in flow_types), \
            f"each element must one of 'P', 'Q' or 'I' but is {flow_types}"
        # id_of_batch[str], P[bool], Q[bool], I[bool]
        selected_batch = batches_with_type[flow_types].apply(any, axis=1)
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
        return split(
            digraph,
            all_edges,
            nodes=nodes_to_hide,
            terminals=terminals_to_hide)
    return make_subgraphs

def get_outputs_of_graph(output_index, edges_of_graph):
    """Collects outputs associated to graph.

    Parameters
    ----------
    model: egrid.model.Model

    edges_of_graph: iterable
        tuples (edges) of connectivity nodes and devices (edges)

    Returns
    -------
    pandas.DataFrame (id_of_batch)
        * .P, bool
        * .Q, bool
        * .I, bool"""
    edges = pd.MultiIndex.from_tuples(
        edges_of_graph, names=['id_of_batch', 'id_of_device'])
    return (
        edges.join(output_index, how='inner')
        .get_level_values(0)
        .drop_duplicates(keep='first'))

def get_make_scaling_of_subgraphs(model):
    """

    Parameters
    ----------
    model: egrid.model.Model
        data of electric power network

    Yields
    ------
    tuple

        * injections, pandas.DataFrame (id)

            * .id_of_node, str
            * .P10, float
            * .Q10, float
            * .Exp_v_p, float
            * .Exp_v_q, float
            * .index_of_node, int
            * .switch_flow_index, int
            * .in_super_node, bool
            * .index_of_terminal, int

        * batches: pandas.DataFrame
            each row represents one batch

            * .id_of_batch
            * .P, bool
            * .Q, bool
            * .I, bool"""
    injections = model.injections.set_index('id')
    outputs = get_outputs(model)
    batches_with_type_ = get_batches_with_type(model, outputs)
    batches_with_type = batches_with_type_.set_index('id_of_batch')
    batches_of_nodes = (
        pd.merge(
            left=outputs
                .drop('id_of_device', axis=1).drop_duplicates(keep='first'),
            left_on='id_of_batch',
            right=batches_with_type,
            right_index=True))
    make_subgraphs = get_make_subgraphs(model, outputs, batches_with_type_)
    output_index = pd.MultiIndex.from_frame(
        outputs[['id_of_batch', 'id_of_device']])
    ids_of_slacks = set(model.slacks.id_of_node)
    has_slack = lambda nodes: bool(ids_of_slacks.intersection(nodes))
    def make_scaling_of_subgraphs(flow_types):
        """Collects injection data and types of flow-measurements per subgraph.

        Parameters
        ----------
        flow_types: list
            of elements 'P', 'Q', 'I'
            addresses flow measurements in particular active power, reactive
            power and magnitude of electric current

        Yields
        ------
        tuple

            * injections, pandas.DataFrame (id)

                * .id_of_node, str
                * .P10, float
                * .Q10, float
                * .Exp_v_p, float
                * .Exp_v_q, float
                * .index_of_node, int
                * .switch_flow_index, int
                * .in_super_node, bool
                * .index_of_terminal, int

            * batches: pandas.DataFrame
                each row represents one batch

                * .id_of_batch
                * .P, bool
                * .Q, bool
                * .I, bool
                
            * has_slack, bool"""
        for edges, nodes, devices in make_subgraphs(flow_types):
            injections_of_subgraph = injections[injections.index.isin(devices)]
            # batches at terminals of subgraphs
            batches_at_terminals = (
                batches_with_type.loc[
                    get_outputs_of_graph(output_index, edges)]
                .reset_index())
            # batches at nodes having outputs in adjacent subgraph
            batches_at_nodes = (
                batches_of_nodes.loc[
                    batches_of_nodes.id_of_node.isin(nodes),
                    batches_at_terminals.columns])
            batches = (
                pd.concat([batches_at_terminals, batches_at_nodes])
                .drop_duplicates(keep='first'))
            yield injections_of_subgraph, batches, has_slack(nodes)
    return make_scaling_of_subgraphs