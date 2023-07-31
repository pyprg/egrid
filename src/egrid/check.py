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
import numpy as np
from egrid._types import (
    SLACKNODES, FACTORS, INJLINKS, INJECTIONS,
    OUTPUTS, IVALUES, PVALUES, QVALUES, VVALUES, BRANCHES, VLIMITS, TERMS)

def check_numbers(frames, msg_cls=(2, 0, 2)):
    """Checks numbers of nodes, injections, and slack nodes.

    Parameters
    ----------
    frames: dict
        * ['Injection'], pandas.DataFrame, injection data
        * ['Slacknode'], pandas.DataFrame, slack data
    msg_cls: array_like<int>
        message levels for
        * 'no slack-node in grid-model'
        * 'no injection in grid-model'
        * 'no node in grid-model'

    Yields
    ------
    tuple
        str, int"""
    assert 3 <= len(msg_cls), 'msg_cls must provide 3 int values'
    count_of_slacknodes = len(frames.get('Slacknode', []))
    count_of_branches = len(frames.get('Branch', BRANCHES))
    count_of_injections = len(frames.get('Injection', []))
    yield f'count of slack-nodes: {count_of_slacknodes}', 0
    yield f'count of branches: {count_of_branches}', 0
    yield f'count of injections: {count_of_injections}', 0
    if count_of_slacknodes < 1:
        yield 'no slack-node in grid-model', msg_cls[0]
    if count_of_injections < 1:
        yield 'no injection in grid-model', msg_cls[1]
    if (count_of_slacknodes + count_of_branches) < 1:
        yield 'no node in grid-model', msg_cls[2]

def check_factor_links(frames, msg_cls=1):
    """Finds factors having no link. Finds links with invalid reference
    to not existing factors/loads.

    Parameters
    ----------
    frames : dict
        * ['Factor'], pandas.DataFrame
        * ['Injectionlink'], pandas.DataFrame
        * ['Injection'], pandas.DataFrame
    msg_cls: int
        class of message

    Yields
    ------
    tuple
        str, int"""
    factors = (
        frames.get('Factor', FACTORS)
        .set_index(['step', 'id'], drop=True))
    injlinks = (
        frames.get('Injectionlink', INJLINKS)
        .set_index(['step', 'id'], drop=True))
    join = factors[['type']].join(injlinks, how="outer")
    for step, id_ in join[join.injid.isna()].index:
        yield (
            f'missing link to load scaling factor \'{id_}\' (step {step})',
            msg_cls)
    for _, row in (
        join[join.type.isna()]
        .reset_index()[['id', 'injid', 'step']]
        .iterrows()):
        yield (
            f'invalid link for injection \'{row.injid}\' in step {row.step}, '
            f'load scaling factor \'{row.id}\' does not exist',
            msg_cls)
    injlinks_ = injlinks.reset_index()
    assoc = injlinks_.copy().set_index(['step', 'injid', 'part'])
    for idx, id_ in assoc[assoc.index.duplicated(keep='first')].iterrows():
        yield f'duplicate Injectionlink (step={idx[0]}, '\
            f'injid=\'{idx[1]}\', part=\'{idx[2]}\'), id=\'{id_[0]}\''
    injections = frames.get('Injection', INJECTIONS)
    valid_inj_ref = injlinks_['injid'].isin(injections.id)
    for _, row in injlinks_[~valid_inj_ref].iterrows():
        yield (
            f'invalid link for load scaling factor \'{row[1]}\' '
            f'in step {row[0]}, injection \'{row[2]}\' does not exist',
            msg_cls)

def check_batch_links(frames, msg_cls=1):
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
    msg_cls: int
        class of message

    Yields
    ------
    tuple
        str, int"""
    output_frame = frames.get('Output', OUTPUTS)
    is_inj_output = output_frame.id_of_node.isna()
    inj_outputs = output_frame[is_inj_output]
    id_of_batch_inj = set(inj_outputs.id_of_batch)
    br_outputs = output_frame[~is_inj_output]
    id_of_batch_br = set(br_outputs.id_of_batch)
    outputs = id_of_batch_br.union(id_of_batch_inj)
    id_of_batch_iv = set(frames.get('IValue', IVALUES).id_of_batch)
    for id_of_batch in id_of_batch_iv - outputs:
        yield (
            f'IValue with invalid id_of_batch reference (\'{id_of_batch}\')',
            msg_cls)
    id_of_batch_pv = set(frames.get('PValue', PVALUES).id_of_batch)
    for id_of_batch in id_of_batch_pv - outputs:
        yield (
            f'PValue with invalid id_of_batch reference (\'{id_of_batch}\')',
            msg_cls)
    id_of_batch_qv = set(frames.get('QValue', QVALUES).id_of_batch)
    for id_of_batch in id_of_batch_qv - outputs:
        yield (
            f'QValue with invalid id_of_batch reference (\'{id_of_batch}\')',
            msg_cls)
    id_of_node_vv = set(frames.get('Vvalue', VVALUES).id_of_node)
    branch_frames = frames.get('Branch', BRANCHES)
    ids_of_nodes_AB = branch_frames[['id_of_node_A', 'id_of_node_B']].stack()
    ids_of_nodes = (
        set(ids_of_nodes_AB)
        .union(frames.get('Slacknode', SLACKNODES).id_of_node))
    for id_of_node in id_of_node_vv - ids_of_nodes:
        yield (
            f'Vvalue with invalid id_of_node reference (\'{id_of_node}\')',
            msg_cls)
    flow_to_batch_refs = (
        id_of_batch_iv.union(id_of_batch_pv).union(id_of_batch_qv))
    for id_of_batch in id_of_batch_br - flow_to_batch_refs:
        yield (
            '(Branch) Output with invalid id_of_batch reference '
            f'(\'{id_of_batch}\')',
            msg_cls)
    for id_of_batch in id_of_batch_inj - flow_to_batch_refs:
        yield (
            '(Injection) Output with invalid id_of_batch reference '
            f'(\'{id_of_batch}\')',
            msg_cls)
    # Outputs with invalid references
    bf = branch_frames[['id', 'id_of_node_A', 'id_of_node_B']]
    if len(bf):
        bf_stacked = bf.set_index('id').stack()
        bf_ids = bf_stacked.index.get_level_values(0)
        idx = pd.MultiIndex.from_tuples(zip(bf_stacked, bf_ids))
    else:
        idx = pd.MultiIndex.from_arrays([[], []])
    br_ = pd.MultiIndex.from_frame(
        br_outputs[['id_of_node', 'id_of_device']])
    is_valid = br_.isin(idx.to_list())
    for _, row in br_outputs[~is_valid].iterrows():
        yield (
            '(Branch) Output with invalid terminal reference '
            f'(id_of_node \'{row.id_of_node}\', '
            f'id_of_branch \'{row.id_of_device}\'), '
            f'id_of_batch \'{row.id_of_batch}\'',
            msg_cls)
    injections = frames.get('Injection', INJECTIONS)
    for _, row in (
            inj_outputs[~inj_outputs.id_of_device.isin(injections.id)]
            .iterrows()):
        yield (
            '(Injection) Output with invalid id_of_injection reference '
            f'(\'{row[1]}\'), id_of_batch \'{row[0]}\'',
            msg_cls)

def check_connections_of_injections(frames, msg_cls=2):
    """Creates messages for disconnected injections.

    Parameters
    ----------
    model_data: dict
        * ['Injection'], pandas.DataFrame
        * ['Branch'], pandas.DataFrame
    msg_cls: int
        class of message

    Yields
    ------
    tuple
        str, int"""
    branch_frames = frames.get('Branch', BRANCHES)
    ids_of_nodes_AB = branch_frames[['id_of_node_A', 'id_of_node_B']].stack()
    ids_of_nodes = (
        set(ids_of_nodes_AB)
        .union(frames.get('Slacknode', SLACKNODES).id_of_node))
    injs = frames.get('Injection', INJECTIONS)
    disconnected_nodes = set(injs.id_of_node) - ids_of_nodes
    for idx, row in injs[injs.id_of_node.isin(disconnected_nodes)].iterrows():
        yield (
            f'disconnected injection id=\'{row.id}\' '\
            f'(id_of_node=\'{row.id_of_node}\')',
            msg_cls)

def check_connections_of_branches(frames, msg_cls=2):
    """Creates messages for disconnected branches.

    Parameters
    ----------
    model_data: dict
        * ['Branch'], pandas.DataFrame
    msg_cls: int
        class of message

    Yields
    ------
    tuple
        str, int"""
    import networkx as nx
    branch_frames = frames.get('Branch', BRANCHES)
    branch_graph = nx.from_pandas_edgelist(
        branch_frames,
        source='id_of_node_A',
        target='id_of_node_B',
        edge_attr='id',
        create_using=None,
        edge_key='id')
    connected_components = [*nx.connected_components(branch_graph)]
    count_of_cc = len(connected_components)
    if 1 < count_of_cc:
        for idx, cc in enumerate(connected_components):
            sg = nx.induced_subgraph(branch_graph, cc)
            ids = list(nx.get_edge_attributes(sg, 'id').values())
            ids.sort()
            yield (
                f'isolated subnetwork {1+idx}/{count_of_cc} '
                f'(branch{"es" if 1 < len(ids) else ""}: {", ".join(ids)})',
                msg_cls)

def check_ids_of_vlimits(frames, msg_cls=1):
    """Checks if Vlimit rows reference existing nodes.

    Parameters
    ----------
    frames: dict
        * ['Node'], pandas.DataFrame with column 'id'
        * ['Vlimit'], pandas.DataFrame with column 'id_of_node'
    msg_cls: int
        class of message

    Yields
    ------
    tuple
        str, int"""
    slack_frame = frames.get('Slacknode', SLACKNODES)
    slack_ids = slack_frame.id_of_node.to_numpy().reshape(-1)
    branch_frame = frames.get('Branch', BRANCHES)
    node_ids = (
        branch_frame[['id_of_node_A','id_of_node_B']].to_numpy().reshape(-1))
    ids = np.concatenate([slack_ids, node_ids])
    vlimits = frames.get('Vlimit', VLIMITS)
    # value '' for id_of_node is a generic value and means for each node
    id_is_valid = (vlimits.id_of_node.isin(ids)) | (vlimits.id_of_node == '')
    for _, row in vlimits[~id_is_valid].iterrows():
        yield(
            f"invalid attribute of Vlimit id_of_node='{row.id_of_node}'",
            msg_cls)

def check_ids_of_terms(frames, msg_cls=1):
    """Checks if Term rows reference existing factors.

    Parameters
    ----------
    frames: dict
        * ['Term'], pandas.DataFrame with column 'id'
        * ['Factor'], pandas.DataFrame with column 'id_of_node'
    msg_cls: int
        class of message

    Yields
    ------
    tuple
        str, int"""
    factorids = frames.get('Factor', FACTORS)[['step','id']].groupby('step')
    def get_invalid(row):
        try:
            ids = set(factorids.get_group(row.step).id)
        except KeyError:
            ids = {}
        return [arg for arg in row.args if arg not in ids]
    terms = frames.get('Term', TERMS)
    invalid_args = terms[['args', 'step']].apply(get_invalid, axis=1)
    for idx, row in terms[invalid_args.astype(bool)].iterrows():
        inv_args = invalid_args[idx]
        pl = 's' if 1 < len(inv_args) else ''
        args = ', '.join(f'\'{arg}\'' for arg in inv_args)
        msg = (
            f"invalid reference{pl} to not existing factor{pl} {args} in "
            f"Term id='{row.id}', fn='{row.fn}', step={row.step}")
        yield msg, msg_cls

def check_ids(frames, msg_cls=2):
    """Checks uniqueness of branch and injection identifiers.

    Issues a message if identifiers are not unique.

    Parameters
    ----------
    frames: dict
        * ['Branch'], pandas.DataFrame with column 'id'
        * ['Injection'], pandas.DataFrame with column 'id'
    msg_cls: int
        class of message

    Yields
    ------
    tuple
        str, int"""
    branches = frames.get('Branch', BRANCHES)
    for idx, row in branches[branches.id.duplicated(keep='first')].iterrows():
        yield f'duplicate branch id=\'{row[0]}\' '\
            f'(id_of_node_A=\'{row[1]}\', id_of_node_B=\'{row[2]}\')'
    injs = frames.get('Injection', INJECTIONS)
    for idx, row in injs[injs.id.duplicated(keep='first')].iterrows():
        yield (
            f'duplicate injection id=\'{row[0]}\' (id_of_node=\'{row[1]}\')',
            msg_cls)

def check_frames(frames):
    """Checks numbers of nodes, injections, and slack nodes. Checks references.

    Finds factors having no link. Finds links with invalid reference
    to not existing factors/loads.
    Finds I/P/Q/Vvalues having an invalid batch/node reference.
    Finds outputs having invalid value reference or device references.
    Finds disconnected injections and branches.
    Finds duplicates in identifiers of injections and branches.
    Finds voltage limits having invalid references to not existing nodes.
    Finds objective function terms having invalid references to not existing
    factors.

    Issues infos(level == 0), warnings (level == 1) and errors (level == 2).

    Creates a pandas DataFrame with:
    ::
        pandas.DataFrame.from_records(
            check_frames(frames),
            columns=['message','level'])

    Parameters
    ----------
    model_data: dict
        * ['Slacknode'], pandas.DataFrame, slack data
        * ['Branch'], pandas.DataFrame
        * ['Injection'], pandas.DataFrame, injection data
        * ['Factor'], pandas.DataFrame
        * ['Injectionlink'], pandas.DataFrame
        * ['IValue'], pandas.DataFrame
        * ['PValue'], pandas.DataFrame
        * ['QValue'], pandas.DataFrame
        * ['Vvalue'], pandas.DataFrame
        * ['Output'], pandas.DataFrame
        * ['Vlimit'], pandas.DataFrame
        * ['Term'], pandas.DataFrame

    Yields
    ------
    tuple
        str, int (message, message_class)"""
    yield from check_numbers(frames)
    yield from check_factor_links(frames)
    yield from check_batch_links(frames)
    yield from check_ids(frames)
    yield from check_connections_of_injections(frames)
    yield from check_connections_of_branches(frames)
    yield from check_ids_of_vlimits(frames)
    yield from check_ids_of_terms(frames)

def get_first_error(frames):
    """Checks if data is usable (free of errors), returns the first error
    message or None.

    Parameters
    ----------
    model_data: dict
        * ['Slacknode'], pandas.DataFrame, slack data
        * ['Branch'], pandas.DataFrame
        * ['Injection'], pandas.DataFrame, injection data
        * ['Factor'], pandas.DataFrame
        * ['Injectionlink'], pandas.DataFrame
        * ['IValue'], pandas.DataFrame
        * ['PValue'], pandas.DataFrame
        * ['QValue'], pandas.DataFrame
        * ['Vvalue'], pandas.DataFrame
        * ['Output'], pandas.DataFrame

    Returs
    ------
    str | None"""
    from itertools import chain
    ch = chain(
        filter(
            lambda msg: msg[1]==2,
            check_numbers(frames, msg_cls=(2, 0, 2))),
        check_connections_of_branches(frames, msg_cls=2),
        check_connections_of_injections(frames, msg_cls=2))
    return next(ch, (None, 0))[0]
