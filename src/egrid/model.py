# -*- coding: utf-8 -*-
"""
Model of an electric distribution network for power flow calculation and
state estimation.

Copyright (C) 2022, 2023 pyprg

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

Created on Sun Aug  19 08:36:10 2021

@author: pyprg
"""
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from collections import namedtuple
from functools import partial
from itertools import chain
from egrid.builder import (
    Slacknode, Branch, Factor, Injectionlink, Terminallink,
    Injection, Output, IValue, PValue, QValue, Vvalue, Vlimit, Term, Message)
from egrid._types import (
    df_astype,
    SLACKNODES, BRANCHES, FACTORS, INJLINKS, TERMINALLINKS,
    INJECTIONS, OUTPUTS, IVALUES, PVALUES, QVALUES, VVALUES, VLIMITS,
    TERMS,
    MESSAGES)
from egrid.factors import make_factordefs, get_factordata_for_step

_Y_LO_ABS_MAX = 1e5

Model = namedtuple(
    'Model',
    'nodes slacks injections terminal_to_branch branchterminals '
    'bridgeterminals '
    'branchoutputs injectionoutputs pvalues qvalues ivalues vvalues vlimits '
    'shape_of_Y count_of_slacks y_max '
    'df_factors injectionlinks terminallinks '
    'mnodeinj terms '
    'messages')
Model.__doc__ = """Data of an electric distribution network.

Model is designed for power flow calculation and state estimation.

Parameters
----------
nodes: pandas.DataFrame (id of node)
    * .idx, int, index of power-flow-calculation node
slacks: pandas.DataFrame
    * .id_of_node, str, id of connection node
    * .V, complex, given voltage at this slack
    * .index_of_node, int, index of power-flow-calculation node
injections: pandas.DataFrame
    * .id, str, unique identifier of injection
    * .id_of_node, str, unique identifier of connected node
    * .P10, float, active power at voltage magnitude 1.0 pu
    * .Q10, float, reactive power at voltage magnitude 1.0 pu
    * .Exp_v_p, float, voltage dependency exponent of active power
    * .Exp_v_q, float, voltage dependency exponent of reactive power
    * .index_of_node, int, index of connected power-flow-calculation node
    * .switch_flow_index, int
    * .in_super_node, bool
    * .is_slack, bool
terminal_to_branch: numpy.array
    * [0] indices of terminal A
    * [1] indices of terminal B
    index of branch is the column index
branchterminals: pandas.DataFrame
    * .index_of_branch, int, index of branch
    * .id_of_branch, str, unique idendifier of branch
    * .id_of_node, str, unique identifier of connected node
    * .id_of_other_node, str, unique identifier of node connected
        at other side of the branch
    * .index_of_node, int, index of connected node
    * .index_of_other_node, int, index of node connected at other side
        of the branch
    * .g_lo, float, longitudinal conductance
    * .b_lo, float, longitudinal susceptance
    * .g_tr_half, float, transversal conductance of branch devided by 2
    * .b_tr_half, float, transversal susceptance pf branch devided by 2
    * .side, str, 'A' | 'B', side of branch, first or second
bridgeterminals: pandas.DataFrame
    terminals of branches being short circuits
    * .index_of_branch, int, index of branch
    * .id_of_branch, str, unique idendifier of branch
    * .id_of_node, str, unique identifier of connected node
    * .id_of_other_node, str, unique identifier of node connected
        at other side of the branch
    * .index_of_node, int, index of connected node
    * .index_of_other_node, int, index of node connected at other side
        of the branch
    * .g_lo, float, longitudinal conductance
    * .b_lo, float, longitudinal susceptance
    * .g_tr_half, float, transversal conductance of branch devided by 2
    * .b_tr_half, float, transversal susceptance pf branch devided by 2
    * .side, str, 'A' | 'B', side of branch, first or second
branchoutputs: pandas.DataFrame
    * .id_of_batch, str, unique identifier of measurement point
    * .id_of_node, str, id of node connected to branch terminal
    * .id_of_branch, str, unique identifier of branch
    * .index_of_node, int, index of node connected to branch terminal
    * .index_of_branch, int, index of branch
injectionoutputs: pandas.DataFrame
    * .id_of_batch, str, unique identifier of measurement point
    * .id_of_injection, str, unique identifier of injection
    * .index_of_injection, str, index of injection
pvalues: pandas.DataFrame
    * .id_of_batch, unique identifier of measurement point
    * .P, float, active power
    * .direction, float, -1: from device into node, 1: from node into device
qvalues: pandas.DataFrame
    * .id_of_batch, unique identifier of measurement point
    * .Q, float, reactive power
    * .direction, float, -1: from device into node, 1: from node into device
ivalues: pandas.DataFrame
    * .id_of_batch, unique identifier of measurement point
    * .I, float, electric current
vvalues: pandas.DataFrame
    * .id_of_node, unique identifier of node voltage is given for
    * .V, float, magnitude of voltage
    * .index_of_node, index of node voltage is given for
vlimits: pandas.DataFrame
    * .id_of_node, str, unique identifier of connectivity node
    * .min, float, smallest value
    * .max, float, greates value
    * .step, int, index of optimization step
    * .index_of_node, index of power-flow-calulation node
shape_of_Y: tuple (int, int)
    shape of admittance matrix for power flow calculation
count_of_slacks: int
    number of slack-nodes for power flow calculation
y_max: float
    * maximum conductance/susceptance value of a branch longitudinal
      admittance, a branch with greater admittance value is regarded
      a connection with inifinite admittance (no impedance), then the
      connectivity nodes of both terminals are aggregated into one
      power-flow-calculation node and the terminals of that branch are
      accessed through Model.bridgeterminals
df_factors: pandas.DataFrame
    * .id, str, unique identifier
    * .type, 'var'|'const', decision variable or parameters
    * .id_of_source, str, id of factor (previous optimization step)
       for initialization
    * .value, float, used by initialization if no source factor in previous
       optimization step
    * .min, float
       smallest possible value
    * .max, float
       greatest possible value
    * .is_discrete, bool
       just 0 digits after decimal point if True, input for solver,
       accepted by MINLP solvers
    * .m, float
       increase of multiplier with respect to change of var/const
       the effective multiplier is a linear function of var/const (mx + n)
    * .n, float
       multiplier when var/const is 0.
       the effective multiplier is a linear function of var/const (mx + n)
    * .step, int, index of optimization step, -1 if all steps
    * .cost, float, cost of change (multiplier for value)
injectionlinks: pandas.DataFrame
    * .injid, str, identifier of injection
    * .part, 'p'|'q', addresses active P or reactive power Q
    * .id, str, identifier of factor
    * .step, int, index of optimization step
terminallinks: pandas.DataFrame
    * .branchid, str, identifier of branch
    * .nodeid, str, identifier of connectivity node
    * .id, str, identifier of factor
    * .step, int, index of optimization step
mnodeinj: scipy.sparse.csc_matrix
    converts a vector ordered according to injection indices to a vector
    ordered according to power flow calculation nodes (adding entries of
    injections for each node) by calculating 'mnodeinj @ vector'
terms: pandas.DataFrame
    * .id, str, unique identifier
    * .args, str, argument for function
    * .fn, str, identifier of function
    * .step, int
messages: pandas.DataFrame
    * .message, str
    * .level, 0 - information, 1 - warning, 2 - error"""

def _join_on(to_join, on_field, dataframe):
    """Joins dataframe with to_join on on_field. Returns a new
    pandas.DataFrame.

    Parameters
    ----------
    to_join: pandas.DataFrame
        * ...
    on_field: str
        name of field to join on
    dataframe: pandas.DataFrame
        * .'on_field'

    Result
    ------
    pandas.DataFrame"""
    return dataframe.join(to_join, on=on_field)

def _add_bg(branches):
    """Prepares data of branches for power flow calculation with seperate real
    and imaginary parts of admittances.

    Parameters
    ----------
    branches: pandas.DataFrame
        * .id
        * .id_of_node_A
        * .id_of_node_B
        * .index_of_node_A
        * .index_of_node_B
        * .y_lo
        * .y_tr

    Returns
    -------
    pandas.DataFrame
        additional columns
            * .id
            * .id_of_node_A
            * .id_of_node_B
            * .index_of_node_A
            * .index_of_node_B
            * .g_lo
            * .b_lo"""
    _branches = branches.copy()
    y_tr_half = branches.y_tr / 2
    _branches['y_tr_half'] = y_tr_half
    _branches['g_tr_half'] = np.real(y_tr_half)
    _branches['b_tr_half'] = np.imag(y_tr_half)
    _branches['g_lo'] = np.real(branches.y_lo)
    _branches['b_lo'] = np.imag(branches.y_lo)
    return _branches.reindex(
        ['id',
         # added for complex calculation
         'y_tr', 'y_tr_half', 'y_lo',
         # end of complex values
         'id_of_node_A', 'id_of_node_B',
         'index_of_node_A', 'index_of_node_B',
         'index_of_term_A', 'index_of_term_B',
         'switch_flow_index_A', 'switch_flow_index_B',
         'g_lo', 'b_lo', 'g_tr_half', 'b_tr_half',
         'index_of_taps_A', 'index_of_taps_B',
         'is_bridge'],
        axis=1)

def _get_branch_terminals(branches, count_of_branches):
    """Prepares data of branch terminals from data of branches.

    Each branch has two terminals. Each branch terminal is connected
    to a node and a branch. The prepared data for a branch terminal provide:

        * data of the connected node, ID and index
        * data of the other node connected to the same branch, ID and index
        * ID of branch

    Terminals of branches which are not short circuits are placed before
    terminals of bridges result.

    Parameters
    ----------
    branches: pandas.DataFrame (index of branch)
        * .id
        * .id_of_node_A
        * .id_of_node_B
        * .index_of_node_A
        * .index_of_node_B
    count_of_branches: int
        number of branches which are not short circuits

    Returns
    -------
    pandas.DataFrame (index of terminal)
        * .id_of_branch
        * .id_of_node
        * .index_of_node
        * .id_of_other_node
        * .index_of_other_node"""
    bras = branches.reset_index().rename(columns={'index': 'index_of_branch'})
    terms_a = (
        bras.rename(
            columns={
                'id'               : 'id_of_branch',
                'id_of_node_A'     : 'id_of_node',
                'index_of_node_A'  : 'index_of_node',
                'index_of_term_A'  : 'index_of_terminal',
                'switch_flow_index_A': 'switch_flow_index',
                'index_of_term_B'  : 'index_of_other_terminal',
                'id_of_node_B'     : 'id_of_other_node',
                'index_of_node_B'  : 'index_of_other_node',
                'index_of_taps_A'  : 'index_of_taps',
                'index_of_taps_B'  : 'index_of_other_taps'})
        .set_index('index_of_terminal')
        .drop('switch_flow_index_B', axis=1))
    terms_a['side_a'] = True
    terms_b = (
        bras.rename(
            columns={
                'id'               : 'id_of_branch',
                'id_of_node_B'     : 'id_of_node',
                'index_of_node_B'  : 'index_of_node',
                'index_of_term_B'  : 'index_of_terminal',
                'switch_flow_index_B': 'switch_flow_index',
                'index_of_term_A'  : 'index_of_other_terminal',
                'id_of_node_A'     : 'id_of_other_node',
                'index_of_node_A'  : 'index_of_other_node',
                'index_of_taps_B'  : 'index_of_taps',
                'index_of_taps_A'  : 'index_of_other_taps'})
        .set_index('index_of_terminal')
        .drop('switch_flow_index_A', axis=1))
    terms_b['side_a'] = False
    return pd.concat([
        terms_a[:count_of_branches],
        terms_b[:count_of_branches],
        terms_a[count_of_branches:],
        terms_b[count_of_branches:]])

# def _prepare_nodes(dataframes):
#     node_ids = np.unique(
#         _getframe(dataframes, Branch, BRANCHES)
#         [['id_of_node_A', 'id_of_node_B']]
#         .to_numpy()
#         .reshape(-1))
#     node_id_index = pd.Index(node_ids, dtype=str)
#     return pd.DataFrame(
#         data={'idx': range(len(node_id_index))},
#         index=node_id_index)

def _prepare_branches(branches, nodes, count_of_branches):
    """Adds IDs of nodes and indices of terminals. Sorts by 'is_bridge'.

    Branches which are not bridges are placed before bridges in the result
    DataFrame.

    Branch-terminals of side A and B are given the first indices. First
    part of branches has also first indices of terminals, hence, the indices
    of branch-terminals can be reused in a terminal-frame. References to them
    are kept consistent in this case.

    Parameters
    ----------
    branches : TYPE
        DESCRIPTION.
    nodes : TYPE
        DESCRIPTION.
    count_of_branches: int
        number of branches which are not short circuits

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    pandas.DataFrame"""
    if not branches['id'].is_unique:
        msg = "Error IDs of branches must be unique but are not."
        raise ValueError(msg)
    nodes_ = nodes[['index_of_node', 'switch_flow_index']]
    branches_ = (
        branches
        .join(nodes_, on='id_of_node_A')
        .join(nodes_, on='id_of_node_B', lsuffix='_A', rsuffix='_B'))
    # first branches then bridges
    branches_.sort_values('is_bridge', ascending=True, inplace=True)
    objectcount = len(branches)
    bridgecount = objectcount - count_of_branches
    branchtermcount = 2 * count_of_branches
    end_of_bridge_term_a = branchtermcount + bridgecount
    termcount = 2 * objectcount
    branches_['index_of_term_A'] = np.concatenate(
        [# branches
         np.arange(count_of_branches, dtype=np.int64),
         # bridges
         np.arange(branchtermcount, end_of_bridge_term_a, dtype=np.int64)])
    branches_['index_of_term_B'] = np.concatenate(
        [# branches
         np.arange(count_of_branches, branchtermcount, dtype=np.int64),
         # bridges
         np.arange(end_of_bridge_term_a, termcount, dtype=np.int64)])
    branches_.reset_index(inplace=True)
    branches_.rename(columns={'index':'index_of_branch'}, inplace=True)
    return branches_

def _prepare_branch_outputs(add_idx_of_node, branches, branchoutputs):
    _branchoutputs = (
        branchoutputs.rename(columns={'id_of_device':'id_of_branch'}))
    # measured branch terminals
    branch_idxs = pd.Series(
        data=branches.index,
        index=branches.id,
        name='index_of_branch')
    return (
        add_idx_of_node(_branchoutputs)
        .join(branch_idxs, on='id_of_branch', how='inner'))

def _prepare_injection_outputs(injections, injectionoutputs):
    _injectionoutputs = (
        injectionoutputs.rename(columns={'id_of_device':'id_of_injection'}))
    # measured injection terminals
    injection_idxs = pd.Series(
        data=injections.index,
        index=injections.id,
        name='index_of_injection')
    return (
        _injectionoutputs
        .join(injection_idxs, on='id_of_injection', how='inner'))

def _get_pfc_nodes(slackids, branch_frame):
    """Collapses nodes connected to impedanceless branches.

    Creates indices for power flow calculation and
    for 'switch flow calculation'. Slack-nodes are returned first.

    Parameters
    ----------
    slackids: pandas.Series
        str, unique identifiers of slack nodes
    branch_frame: pandas.DataFrame

    Returns
    -------
    tuple
        * int, number of power flow slack nodes
        * int, number of power flow calculation nodes
        * pandas.DataFrame (id of node)
            * .index_of_node
            * .switch_flow_index"""
    set_of_slackids = set(slackids)
    get_of_slackids = set_of_slackids.intersection
    is_slack = lambda myset: bool(get_of_slackids(myset))
    # graph connecting all nodes connected to switches and lines having small
    #   impedances
    bridge_graph = nx.from_pandas_edgelist(
        branch_frame[branch_frame.is_bridge],
        source='id_of_node_A',
        target='id_of_node_B',
        edge_attr='id',
        create_using=None,
        edge_key='id')
    bridge_graph.add_nodes_from(slackids)
    connected_components_ = pd.Series(
        nx.connected_components(bridge_graph),
        dtype=object)
    connected_components = pd.DataFrame(
        data={'connected_components': connected_components_,
              'is_slack': connected_components_.apply(is_slack)},
        columns=['connected_components', 'is_slack'])
    cc_count = len(connected_components)
    cc_slacks = (
        connected_components[connected_components.is_slack]
        if cc_count else
        pd.DataFrame([], columns=['connected_components', 'is_slack']))
    cc_slack_count = len(cc_slacks)
    cc_nonslacks = (
        connected_components[~connected_components.is_slack]
        if cc_count else
        pd.DataFrame([], columns=['connected_components', 'is_slack']))
    cc_nonslacks.columns = connected_components.columns
    cc_nodes = set(bridge_graph.nodes)
    # add rest of nodes
    ids_of_branch_nodes = pd.Series(list(
        set(branch_frame
            .loc[~branch_frame.is_bridge,['id_of_node_A', 'id_of_node_B']]
            .to_numpy()
            .reshape(-1))
        - cc_nodes),
        dtype=object)
    branch_nodes = pd.DataFrame(
        data={'id_of_node': ids_of_branch_nodes,
              'is_slack': ids_of_branch_nodes.apply(
                  lambda id_: id_ in set_of_slackids)},
        columns=['id_of_node', 'is_slack'])
    branch_nodes_slacks = (
        branch_nodes[branch_nodes.is_slack]
        if len(branch_nodes) else
        branch_nodes)
    count_of_slacks = cc_slack_count + len(branch_nodes_slacks)
    branch_nodes_nonslacks = (
        branch_nodes[~branch_nodes.is_slack]
        if len(branch_nodes) else
        branch_nodes)
    # 'connected_components' finds groups of nodes connected by switches,
    #   each group will be collapsed to one power flow calculation node
    # all nodes relevant for power flow calculation with indices added
    #   which are usable for matrix building including additional indices
    #   for matrices of switch flow calculation
    return (
        count_of_slacks,
        len(connected_components) + len(branch_nodes),
        pd.DataFrame(
            chain.from_iterable([
                ((id_, idx, switch_flow_index, True, id_ in set_of_slackids)
                  for idx, ids in enumerate(cc_slacks.connected_components)
                  for switch_flow_index, id_ in enumerate(ids)),
                ((id_, idx, 0, False, True)
                  for idx, id_ in enumerate(
                    branch_nodes_slacks.id_of_node, cc_slack_count)),
                ((id_, idx, switch_flow_index, True, False)
                  for idx, ids in enumerate(
                    cc_nonslacks.connected_components, count_of_slacks)
                  for switch_flow_index, id_ in enumerate(ids)),
                ((id_, idx, 0, False, False)
                  for idx, id_ in enumerate(
                    branch_nodes_nonslacks.id_of_node,
                    count_of_slacks + len(cc_nonslacks)))
                ]),
            columns=['node_id', 'index_of_node', 'switch_flow_index',
                     'in_super_node', 'is_slack'])
        .set_index('node_id'))

def get_node_inj_matrix(count_of_nodes, injections):
    """Creates a sparse matrix converting a vector which is ordered
    according to injections to a vector ordered according to power flow
    calculation nodes (adding entries of injections for each node) by
    calculating 'M @ vector'. Transposed M is usable for mapping e.g.
    the vector of node voltage to the vector of injection voltages.

    Parameters
    ----------
    count_of_nodes: int
        number of power flow calculation nodes
    injections: pandas.DataFrame (index of injection)
        * .index_of_node, int

    Returns
    -------
    scipy.sparse.csc_matrix"""
    count_of_injections = len(injections)
    try:
        return coo_matrix(
                ([1] * count_of_injections,
                 (injections.index_of_node, injections.index)),
                shape=(count_of_nodes, count_of_injections),
                dtype=np.int8).tocsc()
    except:
        return coo_matrix(([], ([], [])), shape=(0, 0), dtype=np.int8).tocsc()

def _getframe(frames, cls_, default):
    """Extracts a pandas.DataFrame from frames and returns a copy with
    column types casted to predefined types.

    Parameters
    ----------
    frames: dict
        pandas.DataFrame
    cls_: class
        class of predifined named tuples
    default: pandas.DataFrame
        returned if frames does not provide a DataFrame with
        key cls_.__name__

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    Exception if data cannot be converted into required type"""
    df = frames.get(cls_.__name__)
    if df is None:
        return default
    return df_astype(df, cls_)

def _get_pfc_slacks(slacks):
    slack_groups = slacks.groupby('index_of_node')
    slack_groups_size = slack_groups.size()
    slack_groups_first = (
        slack_groups[['id_of_node', 'switch_flow_index', 'in_super_node']]
        .first())
    slack_groups_first['V'] = slack_groups.V.sum() / slack_groups_size
    return slack_groups_first.reset_index()
    # # identifies slacks connected without impedance to other slacks
    # # currently not used for substitution
    # super_slacks = 1 < slack_groups_size
    # super_slack_idxs = super_slacks[super_slacks].index
    # head_tail = lambda col: (col[0], col[1:].to_list())
    # slackid_groups = [
    #     head_tail(slack_groups.id_of_node.get_group(group_name))
    #     for group_name in super_slack_idxs]

def _get_vlimits(dataframes, pfc_nodes):
    # limits
    vlimits_ = _getframe(dataframes, Vlimit, VLIMITS)
    # connectivity nodes
    empty_node_ids = vlimits_.id_of_node==''
    if any(empty_node_ids):
        node_ids = pfc_nodes.index
        node_count = len(node_ids)
        # complete set of all node Ids for each row with node
        vlimits_empty_ids= vlimits_[empty_node_ids]
        generic = (
            # generate entries for each node per empty id_of_node
            pd.DataFrame(
                np.repeat(
                    vlimits_empty_ids[['min', 'max', 'step']].values,
                    node_count,
                    axis=0),
                columns=['min', 'max', 'step'],
                index=pd.Index(
                    np.concatenate(
                        [node_ids.to_numpy()
                         for _ in range(len(vlimits_empty_ids))]),
                    name='id_of_node'))
            .astype({'step':int})
            .set_index('step', append=True))
        # update with explicitely given values for the specific steps
        generic.update(
            vlimits_[~empty_node_ids].set_index(['id_of_node', 'step']))
        return generic.reset_index(drop=False)
    else:
        return vlimits_

def _aggregate_vlimits(vlimits):
    """Aggregates values for same node and step.
    ::
        min_value = max(min_input_values)
        max_value = min(max_input_values)

    Parameters
    ----------
    vlimits: pandas.DataFrame
        * .index_of_node, int
        * .min, float
           smallest possible magnitude of the voltage at node
        * .max, float
           greatest possible magnitude of the voltage at node
        * .step, int
           index of optimization step, -1 for all steps

    Returns
    -------
    pandas.DataFrame
        * .index_of_node, int
        * .min, float
        * .max, float
        * .step, int"""
    grouped = vlimits.groupby(['step', 'index_of_node'])
    return (
        pd.concat([grouped['min'].max(), grouped['max'].min()], axis=1)
        .reset_index())

def model_from_frames(dataframes=None, y_lo_abs_max=_Y_LO_ABS_MAX):
    """Creates a network model for power flow calculation.

    Parameters
    ----------
    dataframes: dict, str:pandas.DataFrame
        * 'Branch':
            pandas.DataFrame
            * .id, str, ID of branch
            * .id_of_node_A, str, ID of node at terminal A
            * .id_of_node_B, str, ID of node at terminal B
            * .y_lo, complex, longitudinal admittance, pu
            * .y_tr, complex, transversal admittance, pu
        * 'Slacknode':
            pandas.DataFrame
            * .id_of_node, str, ID of slack node
            * .V, complex, given complex voltage, pu
        * 'Injection':
            pandas.DataFrame
            * .id, str, ID of injection
            * .id_of_node, str, ID of connected node
            * .P10, float, active power at voltage 1.0 pu, pu
            * .Q10, float, reactive power at voltage 1.0 pu, pu
            * .Exp_v_p, float, voltage exponent for active power
            * .Exp_v_q, float, voltage exponent for reactive power
        * 'Output':
            pandas.DataFrame
            * .id_of_batch, str
            * .id_of_device, str
            * .id_of_node, str
        * 'PValue':
            pandas.DataFrame
            * .id_of_batch, str
            * .P, float, value of active power, pu
            * .direction, float, -1.0 / 1.0 (out of device / into device)
        * 'QValue':
            pandas.DataFrame
            * .id_of_batch, str
            * .Q, float, value of reactive power, pu
            * .direction, float, -1.0 / 1.0 (out of device / into device)
        * 'IValue':
            pandas.DataFrame
            * .id_of_batch, str
            * .I, float, value of electric current magnitude, pu
        * 'Vvalue':
            pandas.DataFrame
            * .id_of_node, str
            * .V, float, magnitude of voltage, pu
        * 'Factor':
            pandas.DataFrame
            * .step, int, index of estimation step
            * .id, str, ID of factor
            * .type, 'var'|'const', decision variable / constant
            * .id_of_source, str, ID of factor of previous step
            * .value, float, value if no valid initial factor
            * .min, float, lower limit
            * .max, float, upper limit
            * .is_discrete, bool
            * .m, float, increase of multiplier w.r.t. var/const
            * .n, float, multiplier if var/const is 0.
        * 'Injectionlink':
            pandas.DataFrame
            * .step, int, index of estimation step
            * .injid, str, ID of injection
            * .part, 'p'|'q', active/reactive power
            * .id, str, ID of (Load)'Factor'
        * 'Terminallink':
            pandas.DataFrame
            * .step, int, index of estimation step
            * .branchid, str, ID of branch
            * .nodeid, str, ID of connectivity node
            * .id, str, ID of 'Factor'
    y_lo_abs_max: float (default value _Y_LO_ABS_MAX)
        * maximum value of branch longitudinal admittance,
          if the absolute value of the branch admittance is greater
          than y_lo_abs_max it is classified being a bridge
          (connection without impedance)

    Returns
    -------
    Model
        * .nodes, pandas.DataFrame, slack nodes are first
        * .slacks, pandas.DataFrame
        * .injections, pandas.DataFrame
        * .terminal_to_branch, numpy.array, int
        * .branchterminals, pandas.DataFrame
        * .bridgeterminals, pandas.DataFrame
        * .branchoutputs, pandas.DataFrame
        * .injectionoutputs, pandas.DataFrame
        * .pvalues, pandas.DataFrame
        * .qvalues, pandas.DataFrame
        * .ivalues, pandas.DataFrame
        * .vvalues, pandas.DataFrame
        * .vlimits, pandas.DataFrame
        * .shape_of_Y, tuple of int, shape of admittance matrix
        * .count_of_slacks, int, number of slacks for power flow calculation
        * .y_max, float, maximum admittance between two power-flow-calculation
            nodes
        * .factors
        * .mnodeinj
        * .terms
        * .messages"""
    if dataframes is None:
        dataframes = {}
    slacks_ = _getframe(dataframes, Slacknode, SLACKNODES)
    branches_ = _getframe(dataframes, Branch, BRANCHES)
    if not branches_['id'].is_unique:
        msg = "Error: IDs of branches must be unique but are not."
        raise ValueError(msg)
    branches_['is_bridge'] = y_lo_abs_max < branches_.y_lo.abs()
    pfc_slack_count, node_count, pfc_nodes = _get_pfc_nodes(
        slacks_.id_of_node, branches_)
    add_idx_of_node = partial(
        _join_on,
        pfc_nodes[['index_of_node', 'switch_flow_index', 'in_super_node']],
        'id_of_node')
    #  processing of slack nodes
    pfc_slacks = _get_pfc_slacks(
        slacks_.join(pfc_nodes, on='id_of_node', how='inner'))
    # branches and terminals
    count_of_branches = sum(~branches_.is_bridge)
    count_of_branchterms = 2 * count_of_branches
    branches = _prepare_branches(branches_, pfc_nodes, count_of_branches)
    # crossreference branch terminals
    br = branches[:count_of_branches]
    terminal_to_branch = np.vstack([br.index_of_term_A, br.index_of_term_B])
    terminals = _get_branch_terminals(_add_bg(branches), count_of_branches)
    terminals['at_slack'] = (
        terminals.id_of_node.isin(pfc_slacks.id_of_node))
    branchterminals = terminals[:count_of_branchterms]
    branchtermindex = pd.DataFrame(
        {'index_of_terminal': branchterminals.index,
         'index_of_other_terminal':
             branchterminals.index_of_other_terminal.array},
        index=pd.MultiIndex.from_frame(
            branchterminals[['id_of_node', 'id_of_branch']]))
    # injections
    injections_ = add_idx_of_node(_getframe(dataframes, Injection, INJECTIONS))
    if not injections_['id'].is_unique:
        msg = "Error: IDs of injections must be unique but are not."
        raise ValueError(msg)
    first_injindex = 2 * len(branches_)
    last_injindex = first_injindex + len(injections_)
    injections = injections_.assign(
        index_of_terminal=range(first_injindex, last_injindex))
    # limits
    vlimits = add_idx_of_node(_get_vlimits(dataframes, pfc_nodes))
    # measured terminals
    outputs = _getframe(dataframes, Output, OUTPUTS)
    is_branch_output = outputs.id_of_device.isin(branches.id)
    is_injection_output = ~is_branch_output
    branchoutputs = (
        _prepare_branch_outputs(
            add_idx_of_node, branches, outputs[is_branch_output])
        .join(
            branchtermindex['index_of_terminal'],
            on=['id_of_node', 'id_of_branch'],
            how='inner'))
    injectionoutputs = _prepare_injection_outputs(
        injections,
        outputs.loc[is_injection_output, ['id_of_batch', 'id_of_device']])
    # math terms (parts) of objective function
    terms = _getframe(dataframes, Term, TERMS)
    return Model(
        nodes=pfc_nodes,
        slacks=pfc_slacks,
        injections=injections,
        terminal_to_branch=terminal_to_branch,
        branchterminals=branchterminals,
        bridgeterminals=terminals[count_of_branchterms:],
        branchoutputs=branchoutputs,
        injectionoutputs=injectionoutputs,
        pvalues=_getframe(dataframes, PValue, PVALUES),
        qvalues=_getframe(dataframes, QValue, QVALUES),
        ivalues=_getframe(dataframes, IValue, IVALUES),
        vvalues=add_idx_of_node(_getframe(dataframes, Vvalue, VVALUES)),
        vlimits=_aggregate_vlimits(vlimits),
        shape_of_Y=(node_count, node_count),
        count_of_slacks = pfc_slack_count,
        y_max=y_lo_abs_max,
        df_factors=_getframe(dataframes, Factor, FACTORS),
        # factors=_get_factors2(dataframes, branchterminals, injections.id),
        injectionlinks=_getframe(dataframes, Injectionlink, INJLINKS),
        terminallinks=_getframe(dataframes, Terminallink, TERMINALLINKS),
        mnodeinj=get_node_inj_matrix(node_count, injections),
        terms=terms, # data of math terms for objective function
        messages=_getframe(dataframes, Message, MESSAGES.copy()))

def get_pfc_nodes(nodes):
    """Aggregates nodes of same power-flow-calculation node.

    (cim names power-flow-calculation nodes 'topological nodes'
     https://ontology.tno.nl/IEC_CIM/cim_TopologicalNode.html )

    Parameters
    ----------
    nodes: pandas.DataFrame (node_id, str)
        * .index_of_node, int
        * .in_super_node, bool
        * .is_slack, bool

    Returns
    -------
    pandas.DataFrame (node_id, str)
        * .index_of_node, int
        * .is_super_node, bool
        * .is_slack, bool"""
    pfc_nodes_group = nodes.reset_index().groupby('index_of_node')
    pfc_nodes = pfc_nodes_group[['node_id']].first()
    pfc_nodes['is_super_node'] = pfc_nodes_group.in_super_node.any()
    pfc_nodes['is_slack'] = pfc_nodes_group.is_slack.any()
    pfc_nodes.reset_index(inplace=True)
    pfc_nodes.set_index('node_id', inplace=True)
    return pfc_nodes

def _unite(generic, stepspecific):
    index = generic.index.union(stepspecific.index)
    res = generic.reindex(index)
    res.update(stepspecific)
    return res.drop(columns=['step'])

def get_vminmax_for_step(vlimits, index_of_step):
    """Fetches minimum and maximum voltages for index of optimization step.

    Adds generic values to step specific values.

    Parameters
    ----------
    vlimits: pandas.DataFrame
        * .index_of_node, int
        * .min, float
        * .max, float
        * .step, int
    index_of_step : int
        index of optimization step

    Returns
    -------
    pandas.DataFrame (index: index_of_node)
        * .min, float
        * .max, float"""
    generic_limits = vlimits[vlimits.step==-1].set_index('index_of_node')
    step_limits = (
        vlimits[vlimits.step==index_of_step].set_index('index_of_node'))
    return _unite(generic_limits, step_limits)

def get_terms_for_step(terms, index_of_step):
    """Fetches objective function terms for given index of optimization step.

    Adds generic values to step specific values.

    Parameters
    ----------
    terms: pandas.DataFrame
        * .id, int
        * .args, list of str
        * .fn, str
        * .step, int
    index_of_step : int
        index of optimization step

    Returns
    -------
    pandas.DataFrame (index: id)
        * .args, list of str
        * .fn, str"""
    generic_terms = terms[terms.step==-1].set_index('id')
    step_terms = (
        terms[terms.step==index_of_step].set_index('id'))
    return _unite(generic_terms, step_terms)

def _update_positions(factors, pos):
    """Extracts 'value' from factors and overwrites them with matching pos.

    Parameters
    ----------
    factors: egrid.factors.Factors

    pos: iterable
        tuple (id_of_factor, value)

    Returns
    -------
    pandas.Series id_of_factor -> value_of_factor (float)"""
    factorvalues = (
        factors.gen_factordata.loc[
            factors.gen_factordata.index.isin(factors.terminalfactors.id)]
        .value
        .copy())
    positions = (
        pd.DataFrame.from_records(pos, columns=['id', 'value'])
        .set_index('id')
        .value)
    common = factorvalues.index.intersection(positions.index)
    factorvalues[common] = positions[common]
    return factorvalues

def get_positions(factors, pos=()):
    """Returns factors.terminalfactors.value after update with pos.

    The function is intended for manual updates of tap-positions.
    The result can be passed to functions pfcnum.calculate_power_flow
    and pfcnum.calculate_electric_data as argument 'positions'. The return
    value is identical to factors.terminalfactors.value.to_numpy().reshape(-1)
    if pos is not given.

    Parameters
    ----------
    factors: egrid.factors.Factors
        * .gen_factordata
        * .terminalfactors
        factors is retrieved from egrid.model.Model.factors
    pos: iterable, optional
        tuple(str - id_of_terminalfactor, float - position)
        positions for selected terminalfactors,
        example: pos=[('taps', -16), ('taps2', 3)],
        the default is ().

    Returns
    -------
    numpy.ndarray
        values of terminalfactors"""
    return (
        # positions for defined factors
        _update_positions(factors, pos)
        # order value of positions according to factors.terminalfactors
        .reindex(factors.terminalfactors.id)
        .to_numpy()
        .reshape(-1))

def initial_voltage_limits(model):
    """Fetches minimum and maximum voltages for step 0.

    Parameters
    ----------
    vlimits: pandas.DataFrame
        * .index_of_node, int
        * .min, float
        * .max, float
        * .step, int
    index_of_step : int
        index of optimization step

    Returns
    -------
    pandas.DataFrame (index: index_of_node)
        * .min, float
        * .max, float"""
    return get_vminmax_for_step(model.vlimits, 0)

def initial_scaling_factor_values(model):
    """Retrieves initial values for scaling factors (step 0) from model.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric distribution network for calculation

    Returns
    -------
    numpy.array (nx2)
        * [:,0], float, scaling factor for active power,
        * [:,1], float, scaling factor for reactive power"""
    count_of_generic_factors, injs, k, f = get_factordata_for_step(model, 0)
    vals = injs.value
    return np.hstack(
        [vals.iloc[k.kp].to_numpy().reshape(-1,1),
         vals.iloc[k.kq].to_numpy().reshape(-1,1)],
        dtype=np.float64)

def initial_terminal_factor_values(model):
    """Retrieves initial values for terminal factors (step 0) from model.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric distribution network for calculation

    Returns
    -------
    numpy.array"""
    return model.factors.terminalfactors.value.to_numpy().reshape(-1)

def initial_values(model):
    """Retrieves initial value from model.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric distribution network for calculation

    Returns
    -------
    dict
        * ['positions']
        * ['kpq']"""
    return {
        'positions':initial_terminal_factor_values(model),
        'kpq':initial_scaling_factor_values(model)}
