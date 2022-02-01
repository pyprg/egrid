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
from collections import namedtuple
from functools import partial
 
Model = namedtuple(
    'Model',
    'nodes slacks injections branchterminals '
    'branchoutputs injectionoutputs pqvalues ivalues vvalues '
    'branchtaps shape_of_Y slack_indexer '
    'load_scaling_factors injection_factor_association')
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
pqvalues: pandas.DataFrame
    * .id_of_batch, unique identifier of measurement point
    * .P, float, active power
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
    shape of admittance matrix for power flow calculation
slack_indexer: pandas.Series, bool
    True if index is index of slack node, false otherwise"""

def _join_index_of_node(nodes, df):
    return  (
        df
        .join(nodes, on='id_of_node')
        .rename(columns={'idx': 'index_of_node'}))

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
         'g_tot', 'b_tot', 'g_mn', 'b_mn', 'g_mm_half', 'b_mm_half', 
         'index_of_taps_A', 'index_of_taps_B'],
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
    br = branches.reset_index().rename(columns={'index': 'index_of_branch'})
    termsA = (
        br.rename(
            columns={
                'id'             : 'id_of_branch', 
                'id_of_node_A'   : 'id_of_node', 
                'index_of_node_A': 'index_of_node',
                'index_of_term_A': 'index_of_term',
                'index_of_term_B': 'index_of_other_term',
                'id_of_node_B'   : 'id_of_other_node', 
                'index_of_node_B': 'index_of_other_node',
                'index_of_taps_A': 'index_of_taps',
                'index_of_taps_B': 'index_of_other_taps'})
        .set_index('index_of_term'))
    termsA['side'] = 'A'
    termsB = (
        br.rename(
            columns={
                'id'             : 'id_of_branch', 
                'id_of_node_B'   : 'id_of_node', 
                'index_of_node_B': 'index_of_node', 
                'index_of_term_B': 'index_of_term',
                'index_of_term_A': 'index_of_other_term',
                'id_of_node_A'   : 'id_of_other_node', 
                'index_of_node_A': 'index_of_other_node',
                'index_of_taps_B': 'index_of_taps',
                'index_of_taps_A': 'index_of_other_taps'})
        .set_index('index_of_term'))
    termsB['side'] = 'B'
    return pd.concat([termsA, termsB])

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
    is_A = _branchtaps.side == 'A'
    _branchtaps_A = _branchtaps[is_A]
    _branchtaps_B = _branchtaps[~is_A]
    branchtaps_A = (
        _branchtaps_A
        .join(
            _branchtaps_B['id_of_branch']
            .reset_index()
            .rename(columns={'index':'index_of_other_taps'})
            .set_index('id_of_branch'),
            on='id_of_branch'))
    branchtaps_B = (
        _branchtaps_B
        .join(
            _branchtaps_A['id_of_branch']
            .reset_index()
            .rename(columns={'index':'index_of_other_taps'})
            .set_index('id_of_branch'),
            on='id_of_branch'))
    return pd.concat([branchtaps_A, branchtaps_B])

def _prepare_nodes(dataframes):
    node_ids = np.unique(
        dataframes.get('Branch')[['id_of_node_A', 'id_of_node_B']]
        .to_numpy()
        .reshape(-1))
    node_id_index = pd.Index(node_ids, dtype=str)
    return pd.DataFrame(
        data={'idx': range(len(node_id_index))}, 
        index=node_id_index)

def _prepare_branch_taps(add_idx_of_node, dataframes):
    branchtaps = add_idx_of_node(dataframes.get('Branchtaps'))
    branchtaps.reset_index(inplace=True)
    branchtaps.rename(columns={'index':'index_of_taps'}, inplace=True)
    return branchtaps

def _prepare_branches(branchtaps, dataframes, nodes):
    branchtaps_view = (
        branchtaps[['id_of_branch', 'id_of_node', 'index_of_taps']]
        .set_index(['id_of_branch', 'id_of_node']))
    brs = dataframes.get('Branch')
    if not brs['id'].is_unique:
        msg = "Error IDs of branches must be unique but are not."
        raise ValueError(msg)
    _branches = (
        brs
        .join(nodes, on='id_of_node_A')
        .rename(columns={'idx': 'index_of_node_A'})
        .join(nodes, on='id_of_node_B')
        .rename(columns={'idx': 'index_of_node_B'}))
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

def make_model(dataframes):
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
    
    Returns
    -------
    Model
        * .nodes, pandas.DataFrame
        * .slacks, pandas.DataFrame
        * .injections, pandas.DataFrame
        * .branchterminals, pandas.DataFrame
        * .branchoutputs, pandas.DataFrame
        * .injectionoutputs, pandas.DataFrame
        * .pqvalues, pandas.DataFrame
        * .ivalues, pandas.DataFrame
        * .vvalues, pandas.DataFrame
        * .branchtaps, pandas.DataFrame
        * .shape_of_Y, tuple of int, shape of admittance matrix
        * .slack_indexer, pandas.Series
        * .load_scaling_factors
        * .injection_factor_association"""
    nodes = _prepare_nodes(dataframes)
    add_idx_of_node = partial(_join_index_of_node, nodes)
    branchtaps = _prepare_branch_taps(add_idx_of_node, dataframes)
    branches = _prepare_branches(branchtaps, dataframes, nodes)
    slacks = add_idx_of_node(dataframes.get('Slacknode'))
    branchterminals=_get_branch_terminals(_add_bg(branches))
    # injections
    injections = add_idx_of_node(dataframes.get('Injection'))
    if not injections['id'].is_unique:
        msg = "Error IDs of injections must be unique but are not."
        raise ValueError(msg)
    # measured terminals
    outputs = dataframes.get('Output')
    is_branch_output = outputs.id_of_device.isin(branches.id)
    is_injection_output = ~is_branch_output
    branchoutputs = _prepare_branch_outputs(
        add_idx_of_node, branches, outputs[is_branch_output])
    injectionoutputs = _prepare_injection_outputs(
        injections, 
        outputs.loc[is_injection_output, ['id_of_batch', 'id_of_device']])
    size = len(nodes)
    slack_indexer = nodes.idx.isin(slacks.index_of_node)
    load_scaling_factors=dataframes.get('Loadfactor').set_index(['step', 'id'])
    assoc = dataframes.get('KInjlink').set_index(['step', 'injid', 'part'])
    return Model(
        nodes=nodes,
        slacks=slacks, 
        injections=injections,
        branchterminals=branchterminals,
        branchoutputs=branchoutputs,      
        injectionoutputs=injectionoutputs,
        pqvalues=dataframes.get('PQValue'),
        ivalues=dataframes.get('IValue'),
        vvalues=add_idx_of_node(dataframes.get('Vvalue')),
        branchtaps=branchtaps,
        shape_of_Y=(size, size),
        slack_indexer=slack_indexer,
        load_scaling_factors=load_scaling_factors,
        injection_factor_association=assoc)
