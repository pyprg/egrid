# -*- coding: utf-8 -*-
"""
Model of an electric distribution network for power flow calculation and
state estimation.

Copyright (C) 2022 pyprg

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
from collections import namedtuple
from functools import partial
from itertools import chain

Model = namedtuple(
    'Model',
    'nodes slacks injections branchterminals '
    'branchoutputs injectionoutputs pvalues qvalues ivalues vvalues '
    'branchtaps shape_of_Y '
    'load_scaling_factors injection_factor_associations '
    'errormessages')
Model.__doc__ = """Data of an electric distribution network for
power flow calculation and state estimation.

Parameters
----------
nodes: pandas.DataFrame (id of node)
    * .idx, int index of node
slacks: pandas.DataFrame
    * .id_of_node, str, id of connection node
    * .V, complex, given voltage at this slack
    * .index_of_node, int, index of connection node
injections: pandas.DataFrame
    * .id, str, unique identifier of injection
    * .id_of_node, str, unique identifier of connected node
    * .P10, float, active power at voltage magnitude 1.0 pu
    * .Q10, float, reactive power at voltage magnitude 1.0 pu
    * .Exp_v_p, float, voltage dependency exponent of active power
    * .Exp_v_q, float, voltage dependency exponent of reactive power
    * .scalingp, None | str
    * .scalingq, None | str
    * .kp_min, float, minimum of active power scaling factor
    * .kp_max, float, maximum of active power scaling factor
    * .kq_min, float, minimum of reactive power scaling factor
    * .kq_max, float, maximum of reactive power scaling factor
    * .index_of_node, int, index of connected node
branchterminals: pandas.DataFrame
    * .index_of_branch, int, index of branch
    * .id_of_branch, str, unique idendifier of branch
    * .id_of_node, str, unique identifier of connected node
    * .id_of_other_node, str, unique identifier of node connected
        at other side of the branch
    * .index_of_node, int, index of connected node
    * .index_of_other_node, int, index of node connected at other side
        of the branch
    * .g_tot, float, conductance, g_mn + g_mm_half
    * .b_tot, float, susceptande, b_mn + b_mm_half
    * .g_mn, float, longitudinal conductance
    * .b_mn, float, longitudinal susceptance
    * .g_mm_half, float, transversal conductance devided by 2
    * .b_mm_half, float, transversal susceptance devided by 2
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
branchtaps: pandas.DataFrame
    * .id, str, IDs of taps
    * .id_of_node, str, ID of associated node
    * .id_of_branch, str, ID of associated branch
    * .Vstep, float, magnitude of voltage difference per step, pu
    * .positionmin, int, smallest tap position
    * .positionneutral, int, tap with ratio 1:1
    * .positionmax, int, position of greates tap
    * .position, int, actual position
shape_of_Y: tuple (int, int)
    shape of admittance matrix for power flow calculation"""

_EMPTY_TUPLE = ()
_BRANCHES = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id', 'id_of_node_A', 'id_of_node_B', 'y_mn', 'y_mm_half'])
_SLACKNODES = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id_of_node', 'V'])
_INJECTIONS = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id', 'id_of_node', 'P10', 'Q10', 'Exp_v_p', 'Exp_v_q'])
_OUTPUTS = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id_of_batch', 'id_of_device', 'id_of_node'])
_PVALUES = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id_of_batch', 'P', 'direction'])
_QVALUES = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id_of_batch', 'Q', 'direction'])
_IVALUES = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id_of_batch', 'I'])
_VVALUES = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['id_of_node', 'V'])
_BRANCHTAPS = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=[
        'id', 'id_of_node', 'id_of_branch', 'Vstep', 'positionmin',
        'positionneutral', 'positionmax', 'position'])
_LOADFACTORS = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['step', 'id', 'type', 'id_of_source', 'value', 'min', 'max'])
_KINJLINKS = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['step', 'injid', 'part', 'id'])
_KBRANCHLINKS = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['step', 'branchid', 'part', 'id'])
_ERRORMESSAGES = pd.DataFrame(
    _EMPTY_TUPLE,
    columns=['errormessage'])

def _join_index_of_node(nodes, dataframe):
    return dataframe.join(nodes, on='id_of_node')

def _join_index_of_node_inner(nodes, dataframe):
    return dataframe.join(nodes, on='id_of_node', how='inner')

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
        * .y_mn
        * .y_mm_half

    Returns
    -------
    pandas.DataFrame
        additional columns
            * .id
            * .id_of_node_A
            * .id_of_node_B
            * .index_of_node_A
            * .index_of_node_B
            * .y_tot
            * .g_tot
            * .b_tot
            * .g_mn
            * .b_mn"""
    _branches = branches.copy()
    y_tot = branches.y_mn + branches.y_mm_half
    _branches['y_tot'] = y_tot # added for complex calculation
    _branches['g_mm_half'] = np.real(branches.y_mm_half)
    _branches['b_mm_half'] = np.imag(branches.y_mm_half)
    _branches['g_tot'] = np.real(y_tot)
    _branches['b_tot'] = np.imag(y_tot)
    _branches['g_mn'] = np.real(branches.y_mn)
    _branches['b_mn'] = np.imag(branches.y_mn)
    return _branches.reindex(
        ['id',
         # added for complex calculation
         'y_mm_half', 'y_mn', 'y_tot',
         # end of complex values
         'index_of_node_A', 'index_of_node_B',
         'index_of_term_A', 'index_of_term_B',
         'switch_flow_idx_A', 'switch_flow_idx_B',
         'g_tot', 'b_tot', 'g_mn', 'b_mn', 'g_mm_half', 'b_mm_half',
         'index_of_taps_A', 'index_of_taps_B',
         'is_bridge'],
        axis=1)

def _get_branch_terminals(branches):
    """Prepares data of branch terminals from data of branches.
    Each branch has two terminals. Each branch terminal is connected
    to a node and a branch. The prepared data for a branch terminal provide:
        * data of the connected node, ID and index
        * data of the other node connected to the same branch, ID and index
        * ID of branch

    Parameters
    ----------
    branches: pandas.DataFrame (index of branch)
        * .id
        * .id_of_node_A
        * .id_of_node_B
        * .index_of_node_A
        * .index_of_node_B

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
                'index_of_term_A'  : 'index_of_term',
                'switch_flow_idx_A': 'switch_flow_index',
                'index_of_term_B'  : 'index_of_other_term',
                'id_of_node_B'     : 'id_of_other_node',
                'index_of_node_B'  : 'index_of_other_node',
                'index_of_taps_A'  : 'index_of_taps',
                'index_of_taps_B'  : 'index_of_other_taps'})
        .set_index('index_of_term')
        .drop('switch_flow_idx_B', axis=1))
    terms_a['side'] = 'A'
    terms_b = (
        bras.rename(
            columns={
                'id'               : 'id_of_branch',
                'id_of_node_B'     : 'id_of_node',
                'index_of_node_B'  : 'index_of_node',
                'index_of_term_B'  : 'index_of_term',
                'switch_flow_idx_B': 'switch_flow_index',
                'index_of_term_A'  : 'index_of_other_term',
                'id_of_node_A'     : 'id_of_other_node',
                'index_of_node_A'  : 'index_of_other_node',
                'index_of_taps_B'  : 'index_of_taps',
                'index_of_taps_A'  : 'index_of_other_taps'})
        .set_index('index_of_term')
        .drop('switch_flow_idx_A', axis=1))
    terms_b['side'] = 'B'
    return pd.concat([terms_a, terms_b])

def _get_branch_taps_data(branchterminals, tapsframe):
    """Arranges data of taps.

    Parameters
    ----------
    branchterminals: pandas.DataFrame
        * .index_of_branch: str
        * .id_of_branch: str
        * .id_of_node: str
        * .id_of_other_node: str
        * .index_of_node: int
        * .index_of_other_node: int
        * .g_tot: float
        * .b_tot: float
        * .g_mn: float
        * .b_mn: float
        * .g_mm_half: float
        * .b_mm_half: float
        * .side: str
    tapsframe: pandas.DataFrame
        * .id_of_node: str
        * .id_of_branch: str
        * .Vstep: float
        * .positionmin: int
        * .positionneutral: int
        * .positionmax: int
        * .index_of_node: int

    Returns
    -------
    pandas.DataFrame
        * .id_of_node: str
        * .id_of_branch: str
        * .Vstep: float
        * .positionmin: int
        * .positionneutral: int
        * .positionmax: int
        * .index_of_node: int
        * .index_of_terminal: int
        * .side: int
        * .index_of_other_taps: int"""
    # taps of branches
    terminal_indices = (
        branchterminals[['id_of_node', 'id_of_branch', 'side']]
        .reset_index()
        .set_index(['id_of_node', 'id_of_branch'])
        .rename(columns={'index': 'index_of_terminal'}))
    _branchtaps = (
        tapsframe.join(terminal_indices, on=['id_of_node', 'id_of_branch']))
    is_a = _branchtaps.side == 'A'
    _branchtaps_a = _branchtaps[is_a]
    _branchtaps_b = _branchtaps[~is_a]
    branchtaps_a = (
        _branchtaps_a
        .join(
            _branchtaps_b['id_of_branch']
            .reset_index()
            .rename(columns={'index':'index_of_other_taps'})
            .set_index('id_of_branch'),
            on='id_of_branch'))
    branchtaps_b = (
        _branchtaps_b
        .join(
            _branchtaps_a['id_of_branch']
            .reset_index()
            .rename(columns={'index':'index_of_other_taps'})
            .set_index('id_of_branch'),
            on='id_of_branch'))
    return pd.concat([branchtaps_a, branchtaps_b])

_Y_MN_ABS_MAX = 1e5

def _is_short_circuit(y_mn):
    return _Y_MN_ABS_MAX < y_mn.abs

def _prepare_nodes(dataframes):
    node_ids = np.unique(
        dataframes.get('Branch', _BRANCHES)[['id_of_node_A', 'id_of_node_B']]
        .to_numpy()
        .reshape(-1))
    node_id_index = pd.Index(node_ids, dtype=str)
    return pd.DataFrame(
        data={'idx': range(len(node_id_index))},
        index=node_id_index)

def _prepare_branch_taps(add_idx_of_node, dataframes):
    branchtaps = add_idx_of_node(
        dataframes.get('Branchtaps', _BRANCHTAPS))
    branchtaps.reset_index(inplace=True)
    branchtaps.rename(columns={'index':'index_of_taps'}, inplace=True)
    return branchtaps

def _prepare_branches(branchtaps, dataframes, nodes):
    branchtaps_view = (
        branchtaps[['id_of_branch', 'id_of_node', 'index_of_taps']]
        .set_index(['id_of_branch', 'id_of_node']))
    brs = dataframes.get('Branch', _BRANCHES)
    if not brs['id'].is_unique:
        msg = "Error IDs of branches must be unique but are not."
        raise ValueError(msg)
    nodes_ = nodes[['index_of_node', 'switch_flow_idx']]
    _branches = (
        brs
        .join(nodes_, on='id_of_node_A')
        .join(nodes_, on='id_of_node_B', lsuffix='_A', rsuffix='_B'))
    _branches.reset_index(inplace=True)
    _branches.rename(columns={'index':'index_of_branch'}, inplace=True)
    branchcount = len(_branches)
    _branches['index_of_term_A'] = range(branchcount)
    _branches['index_of_term_B'] = range(branchcount, 2 * branchcount)
    return (
        _branches
        .join(branchtaps_view, on=['id', 'id_of_node_A'], how='left')
        .join(branchtaps_view, on=['id', 'id_of_node_B'], how='left',
            lsuffix='_A',
            rsuffix='_B')
        .astype({'index_of_taps_A': 'Int64', 'index_of_taps_B': 'Int64'}))

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

def _get_pfc_nodes(branch_frame):
    """Collapses nodes connected to impedanceless branches.
    Creates indices for power flow calculation and
    for 'switch flow calculation'.

    Parameters
    ----------
    branch_frame: pandas.DataFrame

    Returns
    -------
    tuple
        * int, number of power flow calculation nodes
        * pandas.DataFrame (id of node)
            * .index_of_node
            * .switch_flow_idx"""
    # graph connecting all nodes connected to switches and lines having small
    #   impedances
    bridge_graph = nx.from_pandas_edgelist(
        branch_frame[branch_frame.is_bridge],
        source='id_of_node_A',
        target='id_of_node_B',
        edge_attr='id',
        create_using=None,
        edge_key='id')
    connected_components = [*nx.connected_components(bridge_graph)]
    ccc = len(connected_components)
    cc_nodes = set(bridge_graph.nodes)
    # add rest of nodes
    branch_nodes = set(
        branch_frame
        .loc[~branch_frame.is_bridge,['id_of_node_A', 'id_of_node_B']]
        .to_numpy()
        .reshape(-1))
    # 'connected_components' finds groups of nodes connected by switches,
    #   each group will be collapsed to one power flow calculation node
    # all nodes relevant for power flow calculation with indices added
    #   which are usable for matrix building including additional indices
    #   for matrices of switch flow calculation
    return (
        ccc + len(branch_nodes - cc_nodes),
        pd.DataFrame(
            chain.from_iterable([
                ((id_, idx, switch_flow_idx, True)
                 for idx, ids in enumerate(connected_components)
                 for switch_flow_idx, id_ in enumerate(ids)),
                ((id_, idx, 0, False)
                 for idx, id_ in enumerate(branch_nodes - cc_nodes, ccc))]),
            columns=['node_id', 'index_of_node', 'switch_flow_idx',
                     'in_super_node'])
        .set_index('node_id'))

_EMPTY_DICT = {}

def model_from_frames(dataframes=None, y_mn_abs_max=_Y_MN_ABS_MAX):
    """Creates a network model for power flow calculation.

    Parameters
    ----------
    dataframes: dict, str:pandas.DataFrame
        * 'Branch':
            pandas.DataFrame
            * .id, str, ID of branch
            * .id_of_node_A, str, ID of node at terminal A
            * .id_of_node_B, str, ID of node at terminal B
            * .y_mn, complex, longitudinal admittance, pu
            * .y_mm_half, complex, half of transversal admittance, pu
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
        * 'PQValue':
            pandas.DataFrame
            * .id_of_batch, str
            * .P, float, value of active power, pu
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
        * 'Branchtaps':
            pandas.DataFrame
            * .id, str, IDs of taps
            * .id_of_node, str, ID of associated node
            * .id_of_branch, str, ID of associated branch
            * .Vstep, float, magnitude of voltage difference per step, pu
            * .positionmin, int, smallest tap position
            * .positionneutral, int, tap with ratio 1:1
            * .positionmax, int, position of greates tap
            * .position, int, actual position
        * 'Loadfactor':
            pandas.DataFrame
            * .step, int, index of estimation step
            * .id, str, ID of load factor
            * .type, 'var'|'const', decision variable / constant
            * .id_of_source, str, ID of ini load factor of previous step
            * .value, float, value if no valid initial load factor
            * .min, float, lower limit
            * .max, float, upper limit
        * 'KInjlink':
            pandas.DataFrame
            * .step, int, index of estimation step
            * .injid, str, ID of injection
            * .part, 'p'|'q', active/reactive power
            * .id, str, ID of (Load)factor
        * 'KBranchlink':
            pandas.DataFrame
            * .step, int, index of estimation step
            * .branchid, str, ID of branch
            * .part, 'g'|'b', conductance/susceptance
            * .id, str, ID of branch
    y_mn_abs_max: float (default value _Y_MN_ABS_MAX)
        * maximum value of branch longitudinal admittance,
          if the absolute value of the branch admittance is greater
          than y_mn_abs_max it is classified being a bridge
          (connection without impedance)

    Returns
    -------
    Model
        * .nodes, pandas.DataFrame
        * .slacks, pandas.DataFrame
        * .injections, pandas.DataFrame
        * .branchterminals, pandas.DataFrame
        * .branchoutputs, pandas.DataFrame
        * .injectionoutputs, pandas.DataFrame
        * .pvalues, pandas.DataFrame
        * .qvalues, pandas.DataFrame
        * .pqvalues, pandas.DataFrame
        * .ivalues, pandas.DataFrame
        * .vvalues, pandas.DataFrame
        * .branchtaps, pandas.DataFrame
        * .shape_of_Y, tuple of int, shape of admittance matrix
        * .load_scaling_factors
        * .injection_factor_associations
        * .messages"""
    if not dataframes:
        dataframes = _EMPTY_DICT
    branches_ = dataframes.get('Branch', _BRANCHES)
    branches_['is_bridge'] = y_mn_abs_max < branches_.y_mn.abs()
    size, pfc_nodes = _get_pfc_nodes(branches_)
    if pfc_nodes.empty:
        # create one pfc-node for all injections
        inj = dataframes.get('Injection', _INJECTIONS)
        in_super_node = 2 < inj.shape[0]
        pfc_nodes = pd.DataFrame(
            {'index_of_node':0,
             'switch_flow_idx':0,
             'in_super_node':in_super_node},
            index=inj.id_of_node)
    add_idx_of_node = partial(_join_index_of_node, pfc_nodes)
    #  processing of slack nodes especially if multiple slack-nodes are
    #   placed in the same pfc-node
    slacks = _join_index_of_node_inner(
    pfc_nodes, dataframes.get('Slacknode', _SLACKNODES))
    slack_groups = slacks.groupby('index_of_node')
    slack_groups_size = slack_groups.size()
    slack_groups_first = (
        slack_groups[['id_of_node', 'switch_flow_idx', 'in_super_node']]
        .first())
    slack_groups_first['V'] = slack_groups.V.sum() / slack_groups_size
    pfc_slacks = slack_groups_first.reset_index()
    # identifies slacks connected without impedance to other slacks
    # currently not used for substitution
    super_slacks = 1 < slack_groups_size
    super_slack_idxs = super_slacks[super_slacks].index
    head_tail = lambda col: (col[0], col[1:].to_list())
    slackid_groups = [
        head_tail(slack_groups.id_of_node.get_group(group_name)) 
        for group_name in super_slack_idxs]
    #    
    pfc_nodes['is_slack'] = pfc_nodes.index_of_node.isin(slacks.index_of_node)
    branchtaps = _prepare_branch_taps(add_idx_of_node, dataframes)
    branches = _prepare_branches(branchtaps, dataframes, pfc_nodes)
    branchterminals=_get_branch_terminals(_add_bg(branches))
    # injections
    injections = add_idx_of_node(dataframes.get('Injection', _INJECTIONS))
    if not injections['id'].is_unique:
        msg = "Error: IDs of injections must be unique but are not."
        raise ValueError(msg)
    # measured terminals
    outputs = dataframes.get('Output', _OUTPUTS)
    is_branch_output = outputs.id_of_device.isin(branches.id)
    is_injection_output = ~is_branch_output
    branchoutputs = _prepare_branch_outputs(
        add_idx_of_node, branches, outputs[is_branch_output])
    injectionoutputs = _prepare_injection_outputs(
        injections,
        outputs.loc[is_injection_output, ['id_of_batch', 'id_of_device']])
    load_scaling_factors=(
        dataframes.get('Loadfactor', _LOADFACTORS).set_index(['step', 'id']))
    assoc = (
        dataframes
        .get('KInjlink', _KINJLINKS)
        .set_index(['step', 'injid', 'part']))
    return Model(
        nodes=pfc_nodes,
        slacks=pfc_slacks,
        injections=injections,
        branchterminals=branchterminals,
        branchoutputs=branchoutputs,
        injectionoutputs=injectionoutputs,
        pvalues=dataframes.get('PValue', _PVALUES),
        qvalues=dataframes.get('QValue', _QVALUES),
        ivalues=dataframes.get('IValue', _IVALUES),
        vvalues=add_idx_of_node(dataframes.get('Vvalue', _VVALUES)),
        branchtaps=branchtaps,
        shape_of_Y=(size, size),
        load_scaling_factors=load_scaling_factors,
        injection_factor_associations=assoc,
        errormessages=dataframes.get('errormessages', _ERRORMESSAGES))

def get_pfc_nodes(nodes):
    """Aggregates nodes of same power-flow-calculation node.
    
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
