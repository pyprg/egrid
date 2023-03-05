# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 12:21:32 2023

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
from collections import namedtuple
from itertools import product

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
    active power when magnitude of voltage is 1.0 pu,
    the value is the sum for all three phases
Q10: float, (default value 0)
    reactive power when magnitude of voltage is 1.0 pu
    the value is the sum for all three phases
Exp_v_p: float, (default value 0)
    exponent for voltage dependency of active power,
    0.0 active power P is independent from voltage-magnitude e.g. generators
    2.0 for constant conductance
Exp_v_q: float, (default value 0)
    exponent for voltage dependency of active power,
    0.0 reactive power is independent from voltage-magnitude e.g. generators
    2.0 for constant susceptance"""

Output = namedtuple(
    'Output',
    'id_of_batch id_of_device id_of_node', defaults=(None,))
Output.__doc__ = """Measured terminal or terminal
of a device which is part of a group of devices whose flow is measured.

Parameters
----------
id_of_batch: str
    unique identifier of the batch
id_of_device: str
    id referencing a branch or an injection
id_of_node: str (default value None)
    id of the connected node, 'None' if at injection"""

PValue = namedtuple(
    'PValue',
    'id_of_batch P direction',
    defaults=(0., 1.))
PValue.__doc__ = """Value of (measured) active power. The
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
    active power, sum of all three phases
direction: float (default value 1)
    -1 or 1"""

QValue = namedtuple(
    'QValue',
    'id_of_batch Q direction',
    defaults=(0., 1.))
QValue.__doc__ = """Value of (measured) reactive power. The
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
    reactive power, sum of all three phases
direction: float (default value 1)
    -1 or 1"""

IValue = namedtuple(
    'IValue',
    'id_of_batch I', defaults=(0.0,))
IValue.__doc__ = """Value of (measured) electric current. The
optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Output instance having the identical 'id_of_batch' value
must exist. Placement of multiple measurements in switch fields combined with
several branches or injections are modeled by IValue and Output instances
sharing the same 'id_of_batch'-value.

Parameters
----------
id_of_batch: str
    unique identifier of the point
I: float
    magnitude of electric current, value for one phase"""

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
    magnitude of electric voltage"""

Branchtaps = namedtuple(
    'Branchtaps',
    'id id_of_node id_of_branch Vstep '
    'positionmin positionneutral positionmax position',
    defaults=(10/16, 0, 0, 0, 0))
Branchtaps.__doc__ = """Model of a set of taps.

Parameters
----------
id: str
    unique identifier of taps
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
    used by initialization if no source factor in previous optimization step
min: float (default value numpy.inf)
    smallest possible value
max: float, (default value numpy.inf)
    greatest possible value
step: int (default value 0)
    index of optimization step"""

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
id: str|iterable_of_str
    identifier of scaling factor, unique among factors of same step
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
    greatest value allowed
step: int (default value 0)
    index of optimization step"""

def expand_defk(defk_):
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

def link_(objid, id_, part, cls_, steps):
    """Creates an instance of class cls.

    Parameters
    ----------
    objid: str, or list<str>, or tuple<str>
        id of object to link
    id_: str, or list<str>, or tuple<str>
        id of linked factor, accepts number of parts ids
        (one for 'p' or 'q', two for 'pq')
    part: 'p'|'q'|'pq'
        active power or reactive power
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

Link = namedtuple(
    'Link', 'objid id part cls step', defaults=('pq', KInjlink, 0))
Link.__doc__ = """Logical connection between injection/branch and a scaling
factor.

Parameters
----------
objiid: str|iterable_of_str
    identifier of injection/branch
id: str|iterable_of_str
    identifier of scaling factor to connect, one identifier for each
    given value or argument 'part'
part: 'p'|'q'|'g'|'b'|iterable_of_two_char (default 'pq')
    identifies the attribute of the injection/branch to multipy with factor
    ('p'/'q'- injected active/reactive power, 'g'/'b'- g_lo/b_lo of branch)
cls: KInjlink|KBranchlink (default value KInjlink)
    KInjlink - links an injection
    KBranchlink - links a branch
step: int (default value 0)|iterable_of_int
    addresses the optimization step, first optimization step has index 0"""

Term = namedtuple('Term', 'id arg fn step', defaults=('diff', 0))
Term.__doc__ = """Data of an ojective-function-term.

Parameters
----------
id: str
    unique identifier of a term
arg: str
    reference to an argument of function fn
fn: str
    indentifier of function for calculating a term of the objective function
step: int
    index of optimization step"""

Message = namedtuple('Message', 'message level', defaults=('', 2))
Message.__doc__ = """Information, warning or error message.

Parameters
----------
message: str
    human readable text
level: int
    0 - information
    1 - warning
    2 - error"""

def _tostring(string):
    return string[1:-1] if string.startswith(('\'', '"')) else string

def _notsupported(string):
    def raiseerror(_):
        raise NotImplementedError(string)
    return raiseerror

_ns = (_notsupported
       ('attribute cls is not supported by the text interface, '
        'remove the attribute cls in order to apply the default value'),
       True)
# class => (column_types, function_string_to_type, is_tuple?)
_attribute_types = {
     #    message level
     Message:(
         [object, np.int16],
         [_tostring, np.int16],
         [False, False]),
     #    id      type id_of_source value     min
     #    max         step
     Defk:(
         [object, object, object, np.float64, np.float64,
          np.float64, np.int16],
         [_tostring, _tostring, _tostring,
          np.float64, np.float64, np.float64, np.int16],
         [True, False, False, False, False, False, True]),
     #    id, type, id_of_source, value, min, max, step
     Loadfactor:(
         [object, object, object,
          np.float64, np.float64, np.float64, np.int16],
         [_tostring, _tostring, _tostring,
          np.float64, np.float64, np.float64, np.int16],
         [False, False, False,
          False, False, False, False]),
     #    injid, part, id, step
     KInjlink:(
         [object, object, object, np.int16],
         [_tostring,  _tostring, _tostring, np.int16],
         [False, False, False, False]),
     #    branchid, part, id, step
     KBranchlink:(
         [object, object, object, np.int16],
         [_tostring,  _tostring, _tostring, np.int16],
         [False, False, False, False]),
     #    objid   part    id      cls     step
     Link:(
         [object, object, object, object, np.int16],
         [_tostring,  _tostring, _tostring, _ns, np.int16],
         [True, True, True, False, True]),
     #    id       arg     fn      step
     Term:(
         [object,  object, object, np.int32],
         [_tostring, _tostring, _tostring, np.int32],
         [False, False, False, True]),
     #    id_of_batch id_of_device id_of_node
     Output:(
         [object, object, object],
         [_tostring, _tostring, _tostring],
         [False, False, False]),
     #    id_of_batch   P     direction
     PValue:(
         [object, np.float64, np.float64],
         [_tostring,  np.float64, np.float64],
         [False, False, False]),
     #    id_of_batch   Q     direction
     QValue:(
         [object, np.float64, np.float64],
         [_tostring, np.float64, np.float64],
         [False, False, False]),
     #    id_of_batch   I
     IValue:(
         [object, np.float64],
         [_tostring, np.float64],
         [False, False]),
     #    id_of_node  V
     Vvalue:(
         [object, np.float64],
         [_tostring, np.float64],
         [False, False]),
     Branchtaps:
     #  id   id_of_node id_of_branch Vstep  positionmin positionneutral
     # positionmax position
      ([object, object, object, np.float64, np.int16, np.int16,
        np.int16, np.int16],
       [_tostring, _tostring, _tostring, np.float64, np.int16, np.int16,
       np.int16, np.int16],
       [False, False, False, False, False, False, False, False]),
     #    id      id_of_node_A id_of_node_B y_lo   y_tr
     Branch:(
         [object, object, object, np.float64, np.float64],
         [_tostring, _tostring, _tostring, np.float64, np.float64],
         [False, False, False, False, False]),
     #    id_of_node V
     Slacknode:(
         [object, np.float64],
         [_tostring, np.float64],
         [False, False]),
     #    id     id_of_node  P10      Q10         Exp_v_p     Exp_v_q
     Injection:(
         [object, object, np.float64, np.float64, np.float64, np.float64],
         [_tostring, _tostring, np.float64, np.float64, np.float64,
          np.float64],
         [False, False, False, False, False, False])}

meta_of_types = [
    (cls_, tuple(zip(*info[1:3]))) for cls_, info in _attribute_types.items()]
_EMPTY_TUPLE = ()

def _empty_df(cls_):
    return (
        pd.DataFrame(_EMPTY_TUPLE, columns=cls_._fields)
        .astype(dict(zip(cls_._fields, _attribute_types[cls_][0]))))
# frames with types for columns
SLACKNODES = _empty_df(Slacknode)
BRANCHES = _empty_df(Branch)
INJECTIONS = _empty_df(Injection)
OUTPUTS = _empty_df(Output)
PVALUES = _empty_df(PValue)
QVALUES = _empty_df(QValue)
IVALUES = _empty_df(IValue)
VVALUES = _empty_df(Vvalue)
BRANCHTAPS = _empty_df(Branchtaps)
LOADFACTORS = _empty_df(Loadfactor)
KINJLINKS = _empty_df(KInjlink)
KBRANCHLINKS = _empty_df(KBranchlink)
TERMS = _empty_df(Term)
MESSAGES = _empty_df(Message)
