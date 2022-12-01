# -*- coding: utf-8 -*-
"""
Builds egrid.gridmodel.Model

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

Created on Tue Jan  4 00:21:20 2022

@author: pyprg
"""
from itertools import product, chain
from collections import namedtuple
from functools import singledispatch
import pandas as pd
import numpy as np
import re

Branch = namedtuple(
    'Branch',
    'id id_of_node_A id_of_node_B y_lo y_tr',
    defaults=(complex(np.inf, np.inf), 0.0j))
Branch.__doc__ == """Model of an electrical device having two terminals
e.g. transformers, windings of multi-winding transformers, lines, 
closed switch.

Parameters
----------
id: str
    id of branch
id_of_node_A: str
    id of node at side A
id_of_node_B: str
    id of node at side B
y_lo: complex (default value is complex(numpy.inf, numpy.inf))
    longitudinal admittance
y_tr: complex (default value 0j)
    transversal admittance"""

Slacknode = namedtuple(
    'Slacknode',
    'id_of_node V',
    defaults=(1.+.0j,))
Slacknode.__doc__ = """Tag for a slack node.

Parameters
----------
id_of_node: str
    identifier of the slack node
V: complex (default value 1+0j)
    voltage at slack node"""

Injection = namedtuple(
    'Injection',
    'id id_of_node P10 Q10 Exp_v_p Exp_v_q',
    defaults=(0.0, 0.0, 0.0, 0.0))
Injection.__doc__ = """Model of an electrical one-terminal-device including
consumers (positiv and negative loads), PQ- and PV-generators,
batteries and shunt capacitors.

Parameters
----------
id: str
    unique identifier of injection
id_of_node: str
    id of connected node
P10: float, (default value 0)
    active power at a voltage of 1.0 pu
Q10: float, (default value 0)
    reactive power at a voltage of 1.0 pu
Exp_v_p: float, (default value 0)
    exponent for voltage dependency of active power,
    0.0 if power is independent from voltage-magnitude power e.g. generators
    2.0 for constant conductance
Exp_v_q: float, (default value 0)
    exponent for voltage dependency of active power,
    0.0 power is independent from voltage-magnitude power e.g. generators
    2.0 for constant susceptance"""

Output = namedtuple(
    'Output',
    'id_of_batch id_of_device id_of_node', defaults=(None,))
Output.__doc__ = """Measured terminal or terminal
of a device which is part of a group of devices whose flow is measured.

Parameters
----------
id_of_batch: str
    unique identifier of the point
id_of_device: str
    id referencing a branch or an injection
id_of_node: str (default value None)
    id of the connected node, 'None' if injection"""

PValue = namedtuple(
    'PValue',
    'id_of_batch P direction',
    defaults=(1.,))
PValue.__doc__ = """Values of (measured) active power. The
optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Output instance(s) having the identical 'id_of_batch' value must 
exist. Placement of multiple measurements in switch fields combined with 
several branches or injections are modeled by PValue and Output instances 
sharing the same 'id_of_batch'-value.

Parameters
----------
id_of_batch: str
    unique identifier of the point
P: float
    active power
direction: float (default value 1)
    -1 or 1"""

QValue = namedtuple(
    'QValue',
    'id_of_batch Q direction',
    defaults=(1.,))
QValue.__doc__ = """Values of (measured) reactive power. The
optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Output instance(s) having the identical 'id_of_batch' value 
must exist. Placement of multiple measurements in switch fields combined with 
several branches or injections are modeled by QValue and Output instances 
sharing the same 'id_of_batch'-value.

Parameters
----------
id_of_batch: str
    unique identifier of the point
Q: float
    reactive power
direction: float (default value 1)
    -1 or 1"""

IValue = namedtuple(
    'IValue',
    'id_of_batch I')
IValue.__doc__ = """Values of (measured) electric current. The
optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Branchoutput or Injectionoutput instance having the
identical 'id_of_batch' value must exist. Placement of multiple measurements
in switch fields combined with several branches or injections are modeled
by IValue and Branchoutput/Injectionoutput instances sharing
the same 'id_of_batch'-value.

Parameters
----------
id_of_batch: str
    unique identifier of the point
I: float
    electric current"""

Vvalue = namedtuple(
    'Vvalue',
    'id_of_node V', defaults=(1.,))
Vvalue.__doc__ = """Values of (measured) electric voltage. The
optimization (estimation) target is to meet those (and other given) values.

Parameters
----------
id_of_node: str
    unique identifier of node the voltage was measured at or the
    setpoint is for
Vvalue: float (default value 1.0)
    electric voltage"""

Branchtaps = namedtuple(
    'Branchtaps',
    'id id_of_node id_of_branch Vstep '
    'positionmin positionneutral positionmax position',
    defaults=(10/16, 0, 0, 0, 0))
Branchtaps.__doc__ = """Model of a set of taps.

Parameters
----------
id_of_node: str
    unique identifier of the connected node
id_of_branch: str
    unique identifier of the branch having taps
Vstep: float (default value 10/16)
    increment of voltage per tap step, positive or negative
positionmin: int (default vaLue 0)
    smallest posssible tap postion
positionneutral:int (default vaLue 0)
    position which the voltage is not affected
positionmax: int (default vaLue 0)
    greatest possible tap position
position: int (default vaLue 0)
    actual position"""

DEFAULT_FACTOR_ID = '_default_'

Loadfactor = namedtuple(
    'Loadfactor',
    'id type id_of_source value min max step',
    defaults=('var', DEFAULT_FACTOR_ID, 1.0, -np.inf, np.inf, 0))
Loadfactor.__doc__ = """Data of a load scaling factor.

Parameters
----------
id_: str
    unique idendifier of factor
type: 'var'|'const' (default value 'var')
    decision variable or parameter
id_of_source: str (default value DEFAULT_FACTOR_ID)
    id of scaling factor (previous optimization step) for initialization,
    default value ID of default factor
value: float (default vaLue 1)
    used of initialization if no source factor in previous optimization step,
min: float (default value numpy.inf)
    smalest possible value
max: float, (default value numpy.inf)
    greatest possible value
step: int (default value 0)
    index optimization step"""

def defk(id_, type_='var', id_of_source=None, value=1.0,
          min_=-np.inf, max_=np.inf, step=0):
    """Creates a factor definition for each step.

    Parameters
    ----------
    id_: str
        unique identifier
    type_: 'var'|'const' (default value 'var')
        type of factor decision variable/constant value
    value: float (default value 1)
        used for initialization when no valid given source
    min_: float (default value -numpy.inf)
        smallest possible value
    max_: float (default value numpy.inf)
        greatest possible value
    step: iterable
        int

    Returns
    -------
    pandas.DataFrame"""
    try:
        iter_steps = iter(step)
    except TypeError:
        iter_steps = iter([step])
    ids = id_ if isinstance(id_, (list, tuple)) else [id_]
    return [
        Loadfactor(
            myid_, type_,
            (myid_ if id_of_source is None else id_of_source), 
            value, min_, max_, step_)
        for myid_, step_ in product(ids, iter_steps)]

Defk = namedtuple(
    'Defk',
    'id type id_of_source value min max step',
    defaults=('var', None, 1.0, -np.inf, np.inf, 0))
Defk.__doc__ = """Definition of a scaling factor.

Parameters
----------
id: str
    identifier of scaling factor, unique among factors with same step
type: 'var'|'const' (default value 'var')
    'var' - factor is a decision variable
    'const' - factor is a parameter
id_of_source: str (default value None)
    identifies scaling factor of previous estimation step whose value
    will be used for initialization
value: float (default value 1)
    used for initialization if 'id_of_source' does not reference a
    scaling factor (of previous optimization step)
min: float (default value -numpy.inf)
    smallest value allowed
max: float (default value numpy.inf)
    greates value allowed
step: int (default value 0)
    index of optimization step"""

def _expand_defk(defk_):
    """Creates factor definitions for each step and id.

    Parameters
    ----------
    defk_: Defk

    Returns
    -------
    iterator"""
    try:
        iter_steps = iter(defk_.step)
    except TypeError:
        iter_steps = iter([defk_.step])
    ids = defk_.id if isinstance(defk_.id, (list, tuple)) else [defk_.id]
    return (
        Loadfactor(
            id_, defk_.type,
            (id_ if defk_.id_of_source is None else defk_.id_of_source),
            defk_.value, defk_.min, defk_.max, step_)
        for id_, step_ in product(ids, iter_steps))

KBranchlink = namedtuple('KBranchlink', 'branchid part id step', defaults=(0,))
KBranchlink.__doc__ = """Links branch with scaling factor.

Parameters
----------
branchid: str
    ID of branch
part: 'g'|'b'
    marker for conductance or susceptance
id: str
    unique identifier (for one step) of linked factor
step: int  (default value 0)
    optimization step"""

KInjlink = namedtuple('KInjlink', 'injid part id step', defaults=(0,))
KInjlink.__doc__ = """Links injection with scaling factor.

Parameters
----------
injid: str
    ID of injection
part: 'p'|'q'
    marker for active or reactive power to be scaled
id: str
    unique identifier (for one step) of linked factor
step: int
    optimization step"""

def _link(objid, part, id_, cls_, steps):
    """Creates an instance of class cls.

    Parameters
    ----------
    objid: str, or list<str>, or tuple<str>
        id of object to link
    part: 'p'|'q'
        active power or reactive power
    id_: str, or list<str>, or tuple<str>
        id of linked factor
    cls_: KInjlink
        class of link
    steps: int, or list<int>, or tuple<int>
        index of step"""
    try:
        iter_steps = iter(steps)
    except TypeError:
        iter_steps = iter([steps])
    objids = objid if isinstance(objid, (list, tuple)) else [objid]
    ids = id_ if isinstance(id_, (list, tuple)) else [id_]
    return [cls_(objid_, t[0], t[1], step_, )
            for step_, objid_, t in
                product(iter_steps, objids, zip(part, ids))]

Link = namedtuple('Link', 'objid part id cls step', defaults=(KInjlink, 0))
Link.__doc__ = """Logical connection between injection/branch and a scaling
factor.

Parameters
----------
objiid: str
    identifier of injection/branch
part: 'p'|'q'|'g'|'b'|str
    identifies the attribute of the injection/branch to multipy with factor
    ('p'/'q'- injected active/reactive power, 'g'/'b'- g_lo/b_lo of branch)
id: str
    identifier of scaling factor to connect
cls: KInjlink|KBranchlink (default value KInjlink)
    KInjlink - links a injection
    KBranchlink - links a branch
step: int (default value 0)
    addresses the optimization step, first optimization step has index 0"""

_e3_pattern = re.compile(r'[nuµmkMG]')

_replacement = {
    'n':'e-3', 'u':'e-6', 'µ':'e-6', 'm':'e-3', 
    'k':'e3', 'M':'e6', 'G': 'e9'}

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
    Output, PValue, QValue, IValue, Vvalue, 
    Branchtaps)
SOURCE_TYPES = MODEL_TYPES + (Defk, Link)
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
        return (
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

    Returns
    -------
    Branch"""
    try:
        y_lo = (
            complex(e3(attributes['y_lo'])) 
            if 'y_lo' in attributes else _COMPLEX_INF)
        return Branch(
            id=e_id,
            id_of_node_A=neighbours[0],
            id_of_node_B=neighbours[1],
            y_lo=y_lo,
            y_tr=complex(e3(attributes.get('y_tr', '0.0j'))))
    except KeyError as e:
        return (
            f"Error in data of branch '{e_id}', "
             "for a branch two neighbour nodes are required, "
            f"following neighbours are provided: {str(neighbours)} - "
            f"following attributes are provided: {str(attributes)} "
            f"(error: {str(e)})")
    except ValueError as e:
        return (
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

    Returns
    -------
    Injection|str"""
    # id id_of_node P10 Q10 Exp_v_p Exp_v_q
    if len(neighbours) != 1:
        return (
            f"Error in data of injection '{e_id}', "
            f"the number of neighbours must be exactly 1, "
            f"following neighbours are provided: {str(neighbours)}")
    atts = {}
    for key in ('P10', 'Q10', 'Exp_v_p', 'Exp_v_q'):
        if key in attributes:
            try:
                atts[key] = float(e3(attributes[key]))
            except ValueError as e:
                return (
                    f"Error in data of injection '{e_id}', the value of "
                    f"attribute '{key}' must be of type float if given, "
                    f"following attributes are provided: {str(attributes)} "
                    f"(error: {str(e)})")
    try:
        atts['id'] = e_id
        atts['id_of_node'] = neighbours[0]
        return Injection(**atts)
    except (ValueError, KeyError) as e:
        return (
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

def _create_pvalue(id_of_node, id_of_device, attributes):
    try:
        return True, PValue(
            id_of_batch=f'{id_of_node}_{id_of_device}',
            P=float(e3(attributes['P'])))
    except ValueError as e:
        return False, (
            f"Error in data of edge '{id_of_node}-{id_of_device}', "
             "value of attribute 'P' must be of type float, "
            f"following attributes are provided: {attributes} "
            f"(error: {str(e)})")

def _create_qvalue(id_of_node, id_of_device, attributes):
    try:
        return True, QValue(
            id_of_batch=f'{id_of_node}_{id_of_device}',
            Q=float(e3(attributes['Q'])))
    except ValueError as e:
        return False, (
            f"Error in data of edge '{id_of_node}-{id_of_device}', "
             "value of attribute 'Q' must be of type float, "
            f"following attributes are provided: {attributes} "
            f"(error: {str(e)})")

def _create_ivalue(id_of_node, id_of_device, attributes):
    try:
        return True, IValue(
            id_of_batch=f'{id_of_node}_{id_of_device}',
            I=float(e3(attributes['I'])))
    except ValueError as e:
        return False, (
            f"Error in data of edge '{id_of_node}-{id_of_device}', "
             "value of attribute 'I' must be of type float, "
            f"following attributes are provided: {attributes} "
            f"(error: {str(e)})")

def _make_edge_objects(data):
    """Creates data objects which are associated to an edge.

    Parameters
    ----------
    data: tuple
        str, tuple<str>, dict<str:str>
        ("edge", IDs of connected nodes, attributes)

    Yields
    ------
    PValue | QValue | PQValue | IValue | Output | str"""
    _, neighbours, attributes = data
    if len(neighbours) != 2:
        yield f"edge {neighbours} shall have two nodes"
        return
    a_is_node, b_is_node = (_is_connectivity_node(n) for n in neighbours)
    if a_is_node == b_is_node:
        yield (
            f"Error in data of edge '{neighbours[0]}-{neighbours[1]}', "
             "one node needs to be a connectivity node and one a device node "
             "(IDs of connectivity nodes start with letter 'n', "
             "slack-nodes with prefix 'slack')")
        return
    id_of_node, id_of_device = neighbours if a_is_node else neighbours[::-1]
    create_output = False
    has_p, has_q, has_I = (key in attributes for key in ('P', 'Q', 'I'))
    if has_p:
        success, val = _create_pvalue(id_of_node, id_of_device, attributes)
        yield val
        create_output |= success
    if has_q:
        success, val = _create_qvalue(id_of_node, id_of_device, attributes)
        yield val
        create_output |= success
    if has_I:
        success, val = _create_ivalue(id_of_node, id_of_device, attributes)
        yield val
        create_output |= success
    if create_output:
        yield Output(
            id_of_batch=f'{id_of_node}_{id_of_device}',
            id_of_node=id_of_node,
            id_of_device=id_of_device)

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
                yield(
                    f"Error in data of node '{e_id}', "
                     "value of attribute 'V' must be of type float, "
                    f"following attributes are provided: {attributes} "
                    f"(error: {str(e)})")
                return
    elif count_of_neighbours == 2:
        yield _create_branch(e_id, neighbours, attributes)
    elif count_of_neighbours == 1:
        yield _create_injection(e_id, neighbours, attributes)
    elif count_of_neighbours == 0:
        yield f"ignoring object '{e_id}' as it is not connected"

_FACTORY_FNS = {'edge': _make_edge_objects, 'node': _make_node_objects}
_FACTORY_NONE = lambda x: None

def make_objects(data):
    """Creates objects for edge/node

    Parameters
    ----------
    data: tuple
        * 'edge'|'node'
        * ... ('edge'/'node' specific)

    Returns
    -------
    Branch | Slacknode | Injection | Output | PValue | QValue | IValue | 
    Vvalue | None"""
    return _FACTORY_FNS.get(data[0], _FACTORY_NONE)(data)

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
    return filter(
        lambda x:x, (chain.from_iterable(make_objects(e) for e in entities)))

def make_data_frames(devices):
    """Creates a dictionary of pandas.DataFrame instances from an iterable
    of devices (Branch, Slacknode, Injection, Output, PValue, QValue, IValue, 
    Vvalue, Branchtaps, Defk, Link)

    Parameters
    ----------
    devices: iterable
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
        Branchtaps, Defk, Link

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
            * .id, str, ID of load factor
            * .type, 'var'|'const', decision variable / constant
            * .id_of_source, str, ID of ini load factor of previous step
            * .value, float, value if no valid initial load factor
            * .min, float, lower limit
            * .max, float, upper limit
            * .step, int, index of estimation step
        * 'KInjlink':
            pandas.DataFrame
            * .injid, str, ID of injection
            * .part, 'p'|'q', active/reactive power
            * .id, str, ID of (Load)factor
            * .step, int, index of estimation step
        * 'KBranchlink':
            pandas.DataFrame
            * .branchid, str, ID of branch
            * .part, 'g'|'b', conductance/susceptance
            * .id, str, ID of branch
            * .step, int, index of estimation step
        * 'errormessages':
            pandas.DataFrame
            * .step, int, index of estimation step
            * .branchid, str, ID of branch
            * .part, 'g'|'b', conductance/susceptance
            * .id, str, ID of branch"""
    # collect objects per type
    _slack_and_devs = {src_type.__name__: [] for src_type in _ARG_TYPES}
    for dev in devices:
        _slack_and_devs[type(dev).__name__].append(dev)
    dataframes = {
        dev_type.__name__: pd.DataFrame(
            _slack_and_devs[dev_type.__name__],
            columns=dev_type._fields)
        for dev_type in MODEL_TYPES}
    _factor_frame = pd.DataFrame(
        chain.from_iterable(map(_expand_defk, _slack_and_devs[Defk.__name__])),
        columns=Loadfactor._fields)
    dataframes[Loadfactor.__name__] = _factor_frame
    _injlink_frame = pd.DataFrame(
        chain.from_iterable(
            _link(*args) for args in _slack_and_devs[Link.__name__]
            if args.cls == KInjlink),
        columns=KInjlink._fields)
    dataframes[KInjlink.__name__] = _injlink_frame
    _branchlink_frame = pd.DataFrame(
        chain.from_iterable(
            _link(*args) for args in _slack_and_devs[Link.__name__]
            if args.cls == KBranchlink),
        columns=KBranchlink._fields)
    dataframes[KBranchlink.__name__] = _branchlink_frame
    dataframes['errormessages'] = pd.DataFrame(
        {'errormessages': _slack_and_devs[str.__name__]})
    return dataframes

def _flatten(args):
    if isinstance(args, _ARG_TYPES):
        yield args
    else:
        for arg in args:
            yield from _flatten(arg)

@singledispatch
def _create_objects(arg):
    return arg

@_create_objects.register(str)
def _(arg):
    from graphparser.parsing import parse
    return make_model_objects(parse(arg))

def create_objects(args):
    """Creates instances of network objects from strings. Supports
    engineering notation for floats and complex (see function e3).
    Flattens nested structures. Simply passes network objects to output.

    Parameters
    ----------
    args: iterable
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
        Branchtaps, Defk, Link, str and iterables thereof;
        strings in args are processed with graphparser.parse

    Returns
    -------
    iterator
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue"""
    return _flatten(map(_create_objects, _flatten(args)))    