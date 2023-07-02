# -*- coding: utf-8 -*-
"""
Builds egrid.gridmodel.Model

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

Created on Tue Jan  4 00:21:20 2022

@author: pyprg
"""
import pandas as pd
import numpy as np
import re
from itertools import chain, tee
from collections import defaultdict
from egrid._types import (
    Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
    Vlimit, expand_defvl, Factor, Defk, Deft, Defvl, expand_def,
    DEFAULT_FACTOR_ID,
    Klink, Tlink, expand_klink, expand_tlink, Injectionlink, Terminallink,
    Term, Message, meta_of_types)

_e3_pattern = re.compile(r'[nuµmkMG]')

_replacement = {
    'n':'e-3', 'u':'e-6', 'µ':'e-6', 'm':'e-3', 'k':'e3', 'M':'e6', 'G':'e9'}

def _replace_e3(match):
    """Returns a replacement string for given match.

    Parameters
    ----------
    match: re.match

    Returns
    -------
    str"""
    what = match.group(0)
    return _replacement.get(what, what)

def e3(string):
    """Replaces some letters by e[+|-]n*3.
    'n' -> 'e-9'
    'u' -> 'e-6'
    'µ' -> 'e-6'
    'm' -> 'e-3'
    'k' -> 'e3'
    'M' -> 'e6'
    'G' -> 'e9'

    Parameters
    ----------
    string: str

    Returns
    -------
    str"""
    return re.sub(_e3_pattern, _replace_e3, string)

# all device types of gridmodel.Model and taps and analog values with helper
MODEL_TYPES = (
    Branch, Slacknode, Injection,
    Output, PValue, QValue, IValue, Vvalue, Vlimit,
    Term, Message)
SOURCE_TYPES = MODEL_TYPES + (Defk, Deft, Defvl, Klink, Tlink)
_ARG_TYPES = SOURCE_TYPES + (str,)

def _create_slack(e_id, attributes):
    """Creates a new instance of Slacknode

    Parameters
    ----------
    e_id: str
        ID of element
    attributes: dict

    Returns
    -------
    Slacknode"""
    try:
        voltage = complex(attributes['V'])
        return Slacknode(id_of_node=e_id, V=voltage)
    except KeyError:
        return Slacknode(id_of_node=e_id)
    except ValueError as e:
        return Message(
            f"Error in data of slacknode '{e_id}', "
            f"following attributes are given: {str(attributes)} "
            f"(error: {str(e)})")

_COMPLEX_INF = complex(np.inf, np.inf)

def _create_branch(e_id, neighbours, attributes):
    """Creates a new instance of Branch

    Parameters
    ----------
    e_id: str
        ID of element
    neighbours: tuple
        str, str (ID of left node, ID of right node)
    attributes: dict

    Yields
    ------
    Branch|Message"""
    try:
        y_lo = (
            complex(e3(attributes['y_lo']))
            if 'y_lo' in attributes else _COMPLEX_INF)
        unknown_attributes = attributes.keys() - Branch._fields
        if unknown_attributes:
            be = "is" if len(unknown_attributes) < 2 else "are"
            yield Message(
                f"Error in data of branch '{e_id}', "
                 "unknown attributes, "
                f"following attributes are provided: {str(attributes)}, "
                f"possible attributes are {Branch._fields}, "
                f"hence, {','.join(unknown_attributes)} {be} unknown",
                level=1)
        yield Branch(
            id=e_id,
            id_of_node_A=neighbours[0],
            id_of_node_B=neighbours[1],
            y_lo=y_lo,
            y_tr=complex(e3(attributes.get('y_tr', '0.0j'))))
    except KeyError as e:
        yield Message(
            f"Error in data of branch '{e_id}', "
             "for a branch two neighbour nodes are required, "
            f"following neighbours are provided: {str(neighbours)} - "
            f"following attributes are provided: {str(attributes)} "
            f"(error: {str(e)})")
    except ValueError as e:
        yield Message(
            f"Error in data of branch '{e_id}', "
             "for a branch two neighbour nodes are required, "
             "'y_lo' and 'y_tr' must be complex values , "
            f"following neighbours are provided: {str(neighbours)} - "
            f"following attributes are provided: {str(attributes)} "
            f"(error: {str(e)})")

def _create_injection(e_id, neighbours, attributes):
    """Creates a new instance of Injection. Returns an
    error message if instance cannot be created.

    Parameters
    ----------
    e_id: str
        ID of element
    neighbours: tuple
        str, (ID of node,)
    attributes: dict

    Yields
    ------
    Injection|Message"""
    # id id_of_node P10 Q10 Exp_v_p Exp_v_q
    if len(neighbours) != 1:
        yield Message(
            f"Error in data of injection '{e_id}', "
            f"the number of neighbours must be exactly 1, "
            f"following neighbours are provided: {str(neighbours)}")
        return
    unknown_attributes = attributes.keys() - Injection._fields
    if unknown_attributes:
        be = "is" if len(unknown_attributes) < 2 else "are"
        yield Message(
            f"Error in data of injection '{e_id}', "
             "unknown attributes, "
            f"following attributes are provided: {str(attributes)}, "
            f"possible attributes are {Injection._fields}, "
            f"hence, {','.join(unknown_attributes)} {be} unknown",
            level=1)
    atts = {}
    for key in ('P10', 'Q10', 'Exp_v_p', 'Exp_v_q'):
        if key in attributes:
            try:
                atts[key] = float(e3(attributes[key]))
            except ValueError as e:
                yield Message(
                    f"Error in data of injection '{e_id}', the value of "
                    f"attribute '{key}' must be of type float if given, "
                    f"following attributes are provided: {str(attributes)} "
                    f"(error: {str(e)})")
                return
    try:
        atts['id'] = e_id
        atts['id_of_node'] = neighbours[0]
        yield Injection(**atts)
    except (ValueError, KeyError) as e:
        yield Message(
            f"Error in data of injection, "
             "attributes 'id' and 'id_of_node' are required, "
            f"following attributes are provided: {str(attributes)} "
            f"(error: {str(e)})")

def _is_connectivity_node(string):
    """Checks if node is a connectivity node.

    Parameters
    ----------
    string: str

    Returns
    -------
    bool"""
    return string.startswith('n') or string.startswith('slack')

def _create_pvalue(id_of_node, id_of_device, vals):
    atts = dict(id_of_batch=f'{id_of_node}_{id_of_device}')
    atts.update((k, e3(v)) for k,v in vals.items())
    try:
        return True, PValue(**atts)
    except ValueError as e:
        return False, Message(
            f"Error in data of edge '{id_of_node}-{id_of_device}', "
            f"all values for 'P/P...' ({vals}) "
            f"must be of type float, (error: {str(e)})")

def _create_qvalue(id_of_node, id_of_device, vals):
    atts = dict(id_of_batch=f'{id_of_node}_{id_of_device}')
    atts.update((k, e3(v)) for k,v in vals.items())
    try:
        return True, QValue(**atts)
    except ValueError as e:
        return False, Message(
            f"Error in data of edge '{id_of_node}-{id_of_device}', "
            f"all values for 'Q/Q...' ({vals}) "
            f"must be of type float, (error: {str(e)})")

def _create_ivalue(id_of_node, id_of_device, vals):
    atts = dict(id_of_batch=f'{id_of_node}_{id_of_device}')
    atts.update((k, e3(v)) for k,v in vals.items())
    try:
        return True, IValue(**atts)
    except ValueError as e:
        return False, Message(
            f"Error in data of edge '{id_of_node}-{id_of_device}', "
            f"all values for 'I/I...' ({vals}) "
            f"must be of type float, (error: {str(e)})")

def _collect_attributes(attributes):
    d = defaultdict(dict)
    for key, val in attributes.items():
        m = re.match(r'(\w+)(\.(\w+))?', key)
        if not m is None:
            part_two = m.groups()[2]
            d[m.group(1)][m.string if part_two is None else part_two] = val
    return d

def _make_edge_objects(data):
    """Creates data objects which are associated to an edge.

    Parameters
    ----------
    data: tuple
        str, tuple<str>, dict<str:str>
        ("edge", IDs of connected nodes, attributes)

    Yields
    ------
    PValue | QValue | IValue | Output | str"""
    _, neighbours, attributes = data
    if len(neighbours) != 2:
        yield Message(f"edge {neighbours} shall have two nodes")
        return
    a_is_node, b_is_node = (_is_connectivity_node(n) for n in neighbours)
    if a_is_node == b_is_node:
        yield Message(
            f"Error in data of edge '{neighbours[0]}-{neighbours[1]}', "
             "one node needs to be a connectivity node and one a device node "
             "(IDs of connectivity nodes start with letter 'n', "
             "slack-nodes (which are connectivity nodes) with prefix 'slack')")
        return
    id_of_node, id_of_device = neighbours if a_is_node else neighbours[::-1]
    create_output = False
    collected = _collect_attributes(attributes)
    has_p, has_q, has_I, has_Tl = (
        key in collected for key in ('P', 'Q', 'I', 'Tlink'))
    if has_p:
        success, val = _create_pvalue(
            id_of_node, id_of_device, collected['P'])
        yield val
        create_output |= success
    if has_q:
        success, val = _create_qvalue(
            id_of_node, id_of_device, collected['Q'])
        yield val
        create_output |= success
    if has_I:
        success, val = _create_ivalue(
            id_of_node, id_of_device, collected['I'])
        yield val
        create_output |= success
    if create_output:
        yield Output(
            id_of_batch=f'{id_of_node}_{id_of_device}',
            id_of_node=id_of_node,
            id_of_device=id_of_device)
    if has_Tl:
        terminallink = collected['Tlink'].get('Tlink')
        if terminallink:
            yield Tlink(
                id_of_node=id_of_node,
                id_of_branch=id_of_device,
                id_of_factor=terminallink)

def _make_node_objects(data):
    _, e_id, neighbours, attributes = data
    count_of_neighbours = len(neighbours)
    if _is_connectivity_node(e_id) and count_of_neighbours:
        if attributes.get('slack')=='True' or e_id.startswith('slack'):
            yield _create_slack(e_id, attributes)
        elif 'V' in attributes:
            try:
                yield Vvalue(id_of_node=e_id, V=float(e3(attributes['V'])))
            except ValueError as e:
                yield Message(
                    f"Error in data of node '{e_id}', "
                     "value of attribute 'V' must be of type float, "
                    f"following attributes are provided: {attributes} "
                    f"(error: {str(e)})")
                return
    elif count_of_neighbours == 2:
        yield from _create_branch(e_id, neighbours, attributes)
    elif count_of_neighbours == 1:
        yield from _create_injection(e_id, neighbours, attributes)
    elif count_of_neighbours == 0:
        yield Message(f"ignoring object '{e_id}' as it is not connected", 1)

def _make_nothing(_):
    if False:
        yield None

_FACTORY_FNS = {
    'edge': _make_edge_objects,
    'node': _make_node_objects}

def make_objects(data):
    """Creates objects for edge/node

    Parameters
    ----------
    data: tuple
        * 'edge'|'node'|'comment'
        * ... ('edge'/'node'/'comment' specific)

    Returns
    -------
    Branch | Slacknode | Injection | Output | PValue | QValue | IValue |
    Vvalue | None"""
    return _FACTORY_FNS.get(data[0], _make_nothing)(data)

def make_model_objects(entities):
    """Creates objects from edge/node-tuples.

    Parameters
    ----------
    entities: iterable
        tuples/lists

    Returns
    -------
    iterator
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue"""
    return chain.from_iterable(make_objects(e) for e in entities)

def make_data_frames(devices=()):
    """Creates a dictionary of pandas.DataFrame instances from an iterable
    of devices (Branch, Slacknode, Injection, Output, PValue, QValue, IValue,
    Vvalue, Defk, Deft, Defvl, Klink, Tlink, Message)

    Parameters
    ----------
    devices: iterable, optional
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
        Defk, Deft, Defvl, Kink, Tlink

    Returns
    -------
    dict, str:pandas.DataFrame
        * 'Branch':
            pandas.DataFrame
            * .id, str, ID of branch
            * .id_of_node_A, str, ID of node at terminal A
            * .id_of_node_B, str, ID of node at terminal B
            * .y_lo, complex, longitudinal admittance, pu
            * .y_tr, complex, half of transversal admittance, pu
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
        * 'Vlimit':
            pandas.DataFrame
            * .id_of_node, str
            * .min, float
            * .max, float
            * .step, int
        * 'Factor':
            pandas.DataFrame
            * .id, str, ID of load factor
            * .type, 'var'|'const', decision variable / constant
            * .id_of_source, str, ID of ini load factor of previous step
            * .value, float, value if no valid initial load factor
            * .min, float, lower limit
            * .max, float, upper limit
            * .is_discrete, bool
            * .m, float
            * .n, float
            * .step, int, index of estimation step
        * 'Injectionlink':
            pandas.DataFrame
            * .injid, str, ID of injection
            * .part, 'p'|'q', active/reactive power
            * .id, str, ID of scaling factor
            * .step, int, index of estimation step
        * 'Terminallink':
            pandas.DataFrame
            * .branchid, str, ID of branch
            * .nodeid, str, ID of connectivity node
            * .id, str, ID of taps (terminal) factor
            * .step, int, index of estimation step
        * 'Term': pandas.DataFrame
            * .id, str
            * .step, int
            * .fn, str
            * .arg, str
        * 'Message': pandas.DataFrame
            * .message, str, human readable text
            * .level, int 0 - information, 1 - warning, 2 - error"""
    # collect objects per type
    sources = {src_type.__name__: [] for src_type in _ARG_TYPES}
    for dev in devices:
        if isinstance(dev, _ARG_TYPES):
            sources[type(dev).__name__].append(dev)
        else:
            sources[Message.__name__].append(
                Message(f'wrong type, ignored object: {str(dev)}', 1))
    dataframes = {
        model_type.__name__: pd.DataFrame(
            sources[model_type.__name__],
            columns=model_type._fields)
        for model_type in MODEL_TYPES}
    factor_framek = pd.DataFrame(
        chain.from_iterable(map(expand_def, sources[Defk.__name__])),
        columns=Factor._fields)
    factor_framet = pd.DataFrame(
        chain.from_iterable(map(expand_def, sources[Deft.__name__])),
        columns=Factor._fields)
    dataframes[Factor.__name__] = pd.concat([factor_framek, factor_framet])
    dataframes[Injectionlink.__name__] = pd.DataFrame(
        chain.from_iterable(
            expand_klink(*args) for args in sources[Klink.__name__]),
        columns=Injectionlink._fields)
    dataframes[Terminallink.__name__] = pd.DataFrame(
        chain.from_iterable(
            expand_tlink(*args) for args in sources[Tlink.__name__]),
        columns=Terminallink._fields)
    vlimits = dataframes[Vlimit.__name__]
    vlimits2 = pd.DataFrame(
        chain.from_iterable(
            expand_defvl(defvl) for defvl in sources[Defvl.__name__]),
        columns=Vlimit._fields)
    dataframes[Vlimit.__name__] = pd.concat([vlimits, vlimits2])
    return dataframes

def _flatten(args):
    if isinstance(args, str) or (
            # is named tuple?
            isinstance(args, tuple) and getattr(args, '_fields', None)):
        yield args
    else:
        try:
            for arg in args:
                yield from _flatten(arg)
        except:
            yield Message(f'wrong type, ignored object: {str(args)}', 1)

def _create_objects_from_strings(strings):
    import graphparser as gp
    from graphparser import parse
    type_data = gp.make_type_data(meta_of_types)
    t1, t2 = tee(parse(strings))
    is_comment = lambda t: t[0]=='comment'
    is_instruction = lambda t: t[0]=='comment' and t[1].startswith('#.')
    return chain(
        make_model_objects(t for t in t1 if not is_comment(t)),
        gp.make_objects(
            type_data,
            Message,
            ('  '+t[1][2:] for t in t2 if is_instruction(t))))

def create_objects(args=()):
    """Creates instances of network objects from strings. Supports
    engineering notation for floats and complex (see function e3).
    ::
            'n' -> 'e-9'
            'u' -> 'e-6'
            'µ' -> 'e-6'
            'm' -> 'e-3'
            'k' -> 'e3'
            'M' -> 'e6'
            'G' -> 'e9'

    Flattens nested structures. Simply passes network objects (and others?)
    to output.

    Parameters
    ----------
    args: iterable, optional
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
        Defk, Deft, Defvl, Klink, Tlink, Term, Message, str and
        iterables thereof; strings in args are processed with graphparser.parse

    Returns
    -------
    iterator
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
        Defk, Deft, Defvl, Link, Term, Message"""
    t1, t2 = tee(_flatten(args))
    return chain(
        (t for t in t1 if not isinstance(t, str)),
        _create_objects_from_strings(t for t in t2 if isinstance(t, str)))
