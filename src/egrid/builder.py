# -*- coding: utf-8 -*-
"""
Builds egrid.gridmodel.Model

Copyright (C) 2022  pyprg

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
from itertools import product, chain
from collections import namedtuple
from egrid.gridmodel import make_model
from functools import singledispatch

Branch = namedtuple(
    'Branch', 
    'id id_of_node_A id_of_node_B y_mn y_mm_half')
Branch.__doc__ == """Model of an electrical device having two terminals
e.g. transformer, windings of multi-winding transformers, lines.

Parameters
----------
id: str
    id of branch
id_of_node_A: str
    id of node at side A
id_of_node_B: str
    id of node at side B
y_mn: complex
    longitudinal admittance
y_mm_half: complex
    transversal admittance devided by 2"""

Slacknode = namedtuple(
    'Slacknode', 
    'id_of_node V',
    defaults=(1.+.0j,))
Slacknode.__doc__ = """Tag for a slack node.

Parameters
----------
id_of_node: str
    identifier of the slack node
V: complex
    voltage at slack node, default is 1.0+0.0j"""

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
P10: float
    active power at a voltage of 1.0 pu
Q10: float
    reactive power at a voltage of 1.0 pu
Exp_v_p: float
    exponent for voltage dependency of active power,
    0.0 if power is independent from voltage-magnitude power e.g. generators
    2.0 for constant conductance
Exp_v_q:
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
id_of_node: str
    id of the connected node, 'None' if branch"""
    
PQValue = namedtuple(
    'PQValue',
    'id_of_batch P Q direction', 
    defaults=(1.,))
PQValue.__doc__ = """Values of (measured) active and reactive power. The
optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Branchoutput or Injectionoutput instance having the
identical 'id_of_batch' value must exist. Placement of multiple measurements
in switch fields combined with several branches or injections are modeled
by PQValue and Branchoutput/Injectionoutput instances sharing
the same 'id_of_batch'-value.

Parameters
----------
id_of_batch: str
    unique identifier of the point
P: float 
    active power
Q:float
    reactive power
direction: float
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
    'id_of_node V')
Vvalue.__doc__ = """Values of (measured) electric voltage. The
optimization (estimation) target is to meet those (and other given) values.

Parameters
----------
Vvalue: float
    electric voltage
id_of_node: str
    unique identifier of node the voltage was measured at or the 
    setpoint is for"""

Branchtaps= namedtuple(
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
Vstep: float
    increment of voltage per tap step, positive or negative
positionmin: int
    smallest posssible tap postion
positionneutral:int
    position which the voltage is not affected
positionmax: int
    greatest possible tap position
position: int
    actual position"""

DEFAULT_FACTOR_ID = '_default_'

Loadfactor = namedtuple(
    'Loadfactor', 
    'step id type id_of_source value min max', 
    defaults=('var', DEFAULT_FACTOR_ID, 1.0, -np.inf, np.inf))
Loadfactor.__doc__ = """Data of a load scaling factor.

Parameters
----------
step: int
    index optimization step
id: str
    unique idendifier of factor
type: 'var'|'const'
    decision variable or parameter, default 'var'
id_of_source: str
    id of scaling factor (previous optimization step) for initialization,
    default is ID of default factor
value: float
    used of initialization if no source factor in previous optimization step,
    default 1.0
min: float
    smalest possible value, default -numpy.inf
max: float
    greatest possible value, default numpy.inf"""

def defk(step, id, type='var', id_of_source=None, value=1.0, 
          min=-np.inf, max=np.inf):
    """Creates a factor definition for each step.
    
    Parameters
    ----------
    step: iterable
        int
    
    Returns
    -------
    pandas.DataFrame"""
    try:
        iter_steps = iter(step)
    except:
        iter_steps = iter([step])
    ids = id if isinstance(id, (list, tuple)) else [id]
    return [
        Loadfactor(step_, id_, type, 
            (id_ if id_of_source is None else id_of_source), value, min, max)
        for id_, step_ in product(ids, iter_steps)]

Defk = namedtuple(
    'Defk', 
    'step id type id_of_source value min max', 
    defaults=('var', None, 1.0, -np.inf, np.inf))
Defk.__doc__ = """Definition of a scaling factor.

Parameters
----------
step: int
    index of optimization step
id: str
    identifier of scaling factor, unique among factors with same step
type: 'var'|'const'
    'var' - factor is a decision variable
    'const' - factor is a parameter
id_of_source: str
    identifies scaling factor of previous estimation step whose value
    will be used for initialization
value: float
    used for initialization if 'id_of_source' does not reference a
    scaling factor (of previous optimization step)
min: float
    smallest value allowed
max: float
    greates value allowed"""

def _expand_defk(defk):
    """Creates factor definitions for each step and id.
    
    Parameters
    ----------
    defk: Defk
    
    Returns
    -------
    iterator"""
    try:
        iter_steps = iter(defk.step)
    except:
        iter_steps = iter([defk.step])
    ids = defk.id if isinstance(defk.id, (list, tuple)) else [defk.id]
    return (
        Loadfactor(
            step_, id_, defk.type, 
            (id_ if defk.id_of_source is None else defk.id_of_source),
            defk.value, defk.min, defk.max)
        for id_, step_ in product(ids, iter_steps))
 
KBranchlink = namedtuple('KBranchlink', 'step branchid part id')
KBranchlink.__doc__ = """Links branch with scaling factor.

Parameters
----------
step: int
    optimization step
branchid: str
    ID of branch
part: 'g'|'b'
    marker for conductance or susceptance
id: str
    unique identifier (for one step) of linked factor"""
 
KInjlink = namedtuple('KInjlink', 'step injid part id')
KInjlink.__doc__ = """Links injection with scaling factor.

Parameters
----------
step: int
    optimization step
injid: str
    ID of injection
part: 'p'|'q'
    marker for active or reactive power to be scaled
id: str
    unique identifier (for one step) of linked factor"""

def _link(steps, objid, part, id, cls):
    """Creates an instance of class cls.
    
    Parameters
    ----------
    steps: int, or list<int>, or tuple<int>
        index of step
    objid: str, or list<str>, or tuple<str>
        id of object to link
    part: 'p'|'q'
        active power or reactive power
    id: str, or list<str>, or tuple<str>
        id of linked factor
    cls: KInjlink
        class of link"""
    try:
        iter_steps = iter(steps)
    except:
        iter_steps = iter([steps])
    objids = objid if isinstance(objid, (list, tuple)) else [objid]
    ids = id if isinstance(id, (list, tuple)) else [id]
    return [cls(step_, objid_, t[0], t[1])
            for step_, objid_, t in 
                product(iter_steps, objids, zip(part, ids))]

Link = namedtuple('Link', 'step objid part id cls', defaults=(KInjlink,))

model_types = (
    Branch, Slacknode, Injection, Output, PQValue, IValue, Vvalue, Branchtaps)

source_types = model_types + (Defk, Link)

def _create_slack(e_id, attributes):
    """Creates a new instance of proto.gridmodel.Slacknode
    
    Parameters
    ----------
    e_id: str
        ID of element
    attributes: dict
    
    Returns
    -------
    proto.gridmodel.Slacknode"""
    try:
        voltage = complex(attributes['V'])
        return Slacknode(id_of_node=e_id, V=voltage)
    except:
        return Slacknode(id_of_node=e_id)

def _create_branch(e_id, neighbours, attributes):
    """Creates a new instance of proto.gridmodel.Branch
    
    Parameters
    ----------
    e_id: str
        ID of element
    neighbours: tuple
        str, str (ID of left node, ID of right node)
    attributes: dict
    
    Returns
    -------
    proto.gridmodel.Branch"""
    try:
        return Branch(
            id=e_id, 
            id_of_node_A=neighbours[0], 
            id_of_node_B=neighbours[1],
            y_mn=complex(attributes['y_mn']),
            y_mm_half=complex(attributes['y_mm_half']))
    except:
        msg=(
            f"Error in data of branch '{e_id}', "
             "for a branch two neighbour nodes and two "
             "complex values ('y_mn', 'y_mm_half') are required, "
            f"following neighbours are provided: {str(neighbours)} - "
            f"following attributes are provided: {str(attributes)}")
        raise ValueError(msg)

def _create_injection(e_id, neighbours, attributes):
    """Creates a new instance of proto.gridmodel.Injection.
    
    Parameters
    ----------
    e_id: str
        ID of element
    neighbours: tuple
        str, (ID of node,)
    attributes: dict
    
    Returns
    -------
    proto.gridmodel.Injection"""
    # id id_of_node P10 Q10 Exp_v_p Exp_v_q
    assert len(neighbours) == 1, (
        f"Error in data of injection '{e_id}', "
        f"the number of neighbours must be exactly 1, "
        f"following neighbours are provided: {str(neighbours)}")
    atts = {}
    for key in ('P10', 'Q10', 'Exp_v_p', 'Exp_v_q'):
        if key in attributes:
            try:
                atts[key] = float(attributes[key])
            except:
                msg=(
                    f"Error in data of injection '{e_id}', the value of "
                    f"attribute '{key}' must be of type float if given, "
                    f"following attributes are provided: {str(attributes)}")
                raise ValueError(msg)
    try:
        atts['id'] = e_id
        atts['id_of_node'] = neighbours[0]
        return Injection(**atts)
    except:
        msg=(
            f"Error in data of injection, "
             "attributes 'id' and 'id_of_node' are required, "
            f"following attributes are provided: {str(attributes)}")
        raise ValueError(msg)

def _is_connectivity_node(string):
    return string.startswith('n')

def _make_edge_objects(data):
    """Creates data objects which are associated to an edge.
    
    Parameters
    ----------
    data: tuple
        str, tuple<str>, dict<str:str>
        ("edge", IDs of connected nodes, attributes)
    
    Yields
    ------
    PQValue | IValue | Output
    
    Raises
    ------
    AssertionError"""
    e_type, neighbours, attributes = data
    assert len(neighbours) == 2, f"edge {neighbours} shall have two nodes"
    a_is_node, b_is_node = (_is_connectivity_node(n) for n in neighbours)
    if a_is_node == b_is_node:
        msg = (
            f"Error in data of edge '{neighbours[0]}-{neighbours[1]}', "
             "one node needs to be a connectivity node and one a device node "
             "(the IDs of connectivity nodes start with character 'n')")
    id_of_node, id_of_device = neighbours if a_is_node else neighbours[::-1]
    if 'P' in attributes and 'Q' in attributes:
        id_of_batch = f'PQ_{id_of_node}_{id_of_device}'
        try:
            yield PQValue(
                id_of_batch=id_of_batch,
                P=float(attributes['P']),
                Q=float(attributes['Q']))
        except:
            msg = (
                f"Error in data of edge '{id_of_node}-{id_of_device}', "
                 "values of attributes 'P' and 'Q' must be of type float, "
                f"following attributes are provided: {attributes}")
            raise ValueError(msg)
        yield Output(
            id_of_batch=id_of_batch,
            id_of_node=id_of_node,
            id_of_device=id_of_device)
    if 'I' in attributes:
        id_of_batch = f'I_{id_of_node}_{id_of_device}'
        try:
            yield IValue(
                id_of_batch=id_of_batch,
                I=float(attributes['I']))
        except:
            msg = (
                f"Error in data of edge '{id_of_node}-{id_of_device}', "
                 "value of attribute 'I' must be of type float, "
                f"following attributes are provided: {attributes}")
            raise ValueError(msg)
        yield Output(
            id_of_batch=id_of_batch,
            id_of_node=id_of_node,
            id_of_device=id_of_device)
        
def _make_node_objects(data):
    e_type, e_id, neighbours, attributes = data
    count_of_neighbours = len(neighbours)
    if _is_connectivity_node(e_id) and count_of_neighbours:
        if attributes.get('slack')=='True':
            yield _create_slack(e_id, attributes)
        elif 'V' in attributes:
            try:
                yield Vvalue(
                    id_of_node=e_id,
                    V=float(attributes['V']))
            except:
                msg=(
                    f"Error in data of node '{e_id}', "
                     "value of attribute 'V' must be of type float, "
                    f"following attributes are provided: {attributes}")
                raise ValueError(msg)
    elif count_of_neighbours == 2:
            yield _create_branch(e_id, neighbours, attributes)
    elif count_of_neighbours == 1:
        yield _create_injection(e_id, neighbours, attributes)

_factory_fns = {'edge': _make_edge_objects, 'node': _make_node_objects}
_factory_none = lambda x: None

def make_objects(data):
    return _factory_fns.get(data[0], _factory_none)(data)

def make_model_objects(entities):
    """Creates objects from edge/node-tuples.
    
    Parameters
    ----------
    entities: iterable
        tuples/lists
    
    Returns
    -------
    iterator
        Branch, Slacknode, Injection, Output, PQValue, IValue, Vvalue"""
    return filter(
        lambda x:x, (chain.from_iterable(make_objects(e) for e in entities)))

def make_data_frames(devices):
    """Creates a dictionary of pandas.DataFrame instances from an iterable
    of devices (Branch, Slacknode, Injection, Output, PQValue, IValue, Vvalue, 
    Branchtaps, Defk, Link)
    
    Parameters
    ----------
    devices: iterable
        Branch, Slacknode, Injection, Output, PQValue, IValue, Vvalue, 
        Branchtaps, Defk, Link
    
    Returns
    -------
    dict, str:pandas.DataFrame
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
    
    Raises
    ------
    AssertionError"""
    # collect objects per type
    _slack_and_devs = {src_type.__name__: [] for src_type in source_types}
    for dev in devices:
        _slack_and_devs[type(dev).__name__].append(dev)
    dataframes = {
        dev_type.__name__: pd.DataFrame(
            _slack_and_devs[dev_type.__name__], 
            columns=dev_type._fields)
        for dev_type in model_types}
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
    return dataframes

_arg_types = source_types + (str,)

def _flatten(args):
    if isinstance(args, _arg_types):
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

def get_model(*args):
    """Creates an instance of gridmodel.Model.
    
    Parameters
    ----------
    args: iterable
        Branch, Slacknode, Injection, Output, PQValue, IValue, Vvalue, 
        Branchtaps, Defk, Link, str
        strings in args are processed with argparser.parsing.parse
    
    Returns
    -------
    gridmodel.Model"""
    frames = make_data_frames(_flatten(map(_create_objects, _flatten(args))))
    return make_model(frames)
