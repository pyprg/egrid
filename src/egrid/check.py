# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 12:18:54 2023

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

@author: pyprg
"""

import pandas as pd
from src.egrid.input import (
    SLACKNODES, LOADFACTORS, KINJLINKS, INJECTIONS,
    OUTPUTS, IVALUES, PVALUES, QVALUES, VVALUES, BRANCHES)

def check_numbers(model_data):
    """Checks numbers of nodes, injections, and slack nodes.
    
    Parameters
    ----------
    model_data: dict
        * ['shape_of_Y'] tuple<int>, shape of branch admittance matrix
        * ['Injection'], pandas.DataFrame, injection data
        * ['Slacknode'], pandas.DataFrame, slack data
    
    Yields
    ------
    str"""
    if model_data.get('shape_of_Y', (0,0))[0] < 1:
        yield 'no node in grid-model'
    if len(model_data.get('Injection', [])) < 1:
        yield 'no injection in grid-model'
    if len(model_data.get('Slacknode', [])) < 1:
        yield 'no slack-node in grid-model'

def check_factor_links(frames):
    """Finds factors having no link. Finds links with invalid reference
    to not existing factors/loads.
    
    Parameters
    ----------
    frames : dict
        * ['Loadfactor'], pandas.DataFrame
        * ['KInjlink'], pandas.DataFrame
        * ['Injection'], pandas.DataFrame

    Yields
    -------
    str"""
    factors = (
        frames.get('Loadfactor', LOADFACTORS)
        .set_index(['step', 'id'], drop=True))
    kinjlinks = (
        frames.get('KInjlink', KINJLINKS)
        .set_index(['step', 'id'], drop=True))
    join = factors[['type']].join(kinjlinks, how="outer")
    for step, id_ in join[join.injid.isna()].index:
        yield f'missing link to load scaling factor \'{id_}\' (step {step})'
    for _, row in (
        join[join.type.isna()]
        .reset_index()[['id', 'injid', 'step']]
        .iterrows()):
        yield f'invalid link for injection \'{row[1]}\' in step {row[2]}, '\
            f'load scaling factor \'{row[0]}\' does not exist'
    kinjlinks_ = kinjlinks.reset_index()
    injections = frames.get('Injection', INJECTIONS)
    valid_inj_ref = kinjlinks_['injid'].isin(injections.id)
    for _, row in kinjlinks_[~valid_inj_ref].iterrows():
        yield f'invalid link for load scaling factor \'{row[1]}\' '\
            f'in step step {row[0]}, injection \'{row[2]}\' does not exist'

def check_batch_links(frames):
    """Finds I/P/Q/Vvalues having an invalid batch/node reference. 
    Finds outputs having invalid value or device references.
    
    Parameters
    ----------
    frames : dict
        * ['Slacknode'], pandas.DataFrame
        * ['Branch'], pandas.DataFrame
        * ['Injection'], pandas.DataFrame
        * ['IValue'], pandas.DataFrame
        * ['PValue'], pandas.DataFrame
        * ['QValue'], pandas.DataFrame
        * ['Vvalue'], pandas.DataFrame
        * ['Output'], pandas.DataFrame

    Yields
    -------
    str"""
    output_frame = frames.get('Output', OUTPUTS)
    is_inj_output = output_frame.id_of_node.isna()
    inj_outputs = output_frame[is_inj_output]
    id_of_batch_inj = set(inj_outputs.id_of_batch)
    br_outputs = output_frame[~is_inj_output]
    id_of_batch_br = set(br_outputs.id_of_batch)
    outputs = id_of_batch_br.union(id_of_batch_inj)
    id_of_batch_iv = set(frames.get('IValue', IVALUES).id_of_batch)
    for id_of_batch in id_of_batch_iv - outputs:
        yield f'IValue with invalid id_of_batch reference (\'{id_of_batch}\')'
    id_of_batch_pv = set(frames.get('PValue', PVALUES).id_of_batch)
    for id_of_batch in id_of_batch_pv - outputs:
        yield f'PValue with invalid id_of_batch reference (\'{id_of_batch}\')'
    id_of_batch_qv = set(frames.get('QValue', QVALUES).id_of_batch)
    for id_of_batch in id_of_batch_qv - outputs:
        yield f'QValue with invalid id_of_batch reference (\'{id_of_batch}\')'
    id_of_node_vv = set(frames.get('Vvalue', VVALUES).id_of_node)
    branch_frames = frames.get('Branch', BRANCHES)
    ids_of_nodes_AB = branch_frames[['id_of_node_A', 'id_of_node_B']].stack()
    ids_of_nodes = (
        set(ids_of_nodes_AB)
        .union(frames.get('Slacknode', SLACKNODES).id_of_node))
    for id_of_node in id_of_node_vv - ids_of_nodes:
        yield f'Vvalue with invalid id_of_node reference (\'{id_of_node}\')'
    flow_to_batch_refs = (
        id_of_batch_iv.union(id_of_batch_pv).union(id_of_batch_qv))
    for id_of_batch in id_of_batch_br - flow_to_batch_refs:
        yield '(Branch) Output with invalid id_of_batch reference '\
            f'(\'{id_of_batch}\')'
    for id_of_batch in id_of_batch_inj - flow_to_batch_refs:
        yield '(Injection) Output with invalid id_of_batch reference '\
            f'(\'{id_of_batch}\')'
    # Outputs with invalid references
    ids_of_branches_ = branch_frames.id
    ids_of_branches = pd.concat([ids_of_branches_, ids_of_branches_])
    idx = pd.MultiIndex.from_tuples(
        zip(ids_of_nodes_AB, ids_of_branches), 
        names=['id_of_node', 'id_of_device'])
    branch_terms = pd.Series(True, idx, name='exists')
    br_ = (
        br_outputs[['id_of_node', 'id_of_device', 'id_of_batch']]
        .set_index(['id_of_node', 'id_of_device']))
    for _, row in (
            br_[br_.join(branch_terms,how='left').exists.isna()]
            .reset_index()
            .iterrows()):
        yield '(Branch) Output with invalid terminal reference '\
            f'(id_of_node \'{row[0]}\', id_of_branch \'{row[1]}\'), '\
            f'id_of_batch \'{row[2]}\''
    injections = frames.get('Injection', INJECTIONS)
    for _, row in (
            inj_outputs[~inj_outputs.id_of_device.isin(injections.id)]
            .iterrows()):
        yield '(Injection) Output with invalid id_of_injection reference '\
            f'(\'{row[1]}\'), id_of_batch \'{row[0]}\''
