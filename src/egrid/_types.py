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

DEFAULT_FACTOR_ID = '_default_'

Factor = namedtuple(
    'Factor',
    'id type id_of_source value min max is_discrete m n step',
    defaults=(
        'var', DEFAULT_FACTOR_ID, 1.0, -np.inf, np.inf, False, 1., 0., -1))
Factor.__doc__ = """Data of a factor.

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
is_discrete: bool (default is False)
    no values after decimal point if True, input for solver accepted
    by MINLP solvers
m: float (default 1.)
    dy/dx, effective multiplier is a linear function f(x) = mx + n, m is the
    increase of that linear function
n: float (default 0.)
    effective multiplier is a linear function f(x) = mx + n, n is f(0)
step: int (default value -1)
    index of optimization step, defined for each step if set to -1"""

Defk = namedtuple(
    'Defk',
    'id type id_of_source value min max is_discrete m n step',
    defaults=('var', None, 1.0, -np.inf, np.inf, False, 1., 0., -1))
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
is_discrete: bool (default values is False)
    input for MINLP solver, indicates if factor shall be processed like int
m: float (default 1.)
    dy/dx, effective multiplier is a linear function f(x) = mx + n,
    m is the increase of that linear function
    not used
n: float (default 0.)
    effective multiplier is a linear function f(x) = mx + n, n is f(0)
    not used
step: int (default value -1)
    index of optimization step, for each step set to -1"""

Deft = namedtuple(
    'Deft',
    'id type id_of_source value min max is_discrete m n step',
    defaults=('const', None, 0., -16., 16., True, -.1/16, 1., -1))
Deft.__doc__ = """Definition of a taps (terminal) factor.

Parameters
----------
id: str|iterable_of_str
    identifier of scaling factor, unique among factors of same step
type: 'var'|'const' (default value 'const')
    'var' - factor is a decision variable
    'const' - factor is a parameter
id_of_source: str (default value None)
    identifies scaling factor of previous estimation step whose value
    will be used for initialization
value: float (default value 0)
    used for initialization if 'id_of_source' does not reference a
    scaling factor (of previous optimization step)
min: float (default value -16)
    smallest value allowed
max: float (default 16)
    greatest value allowed
is_discrete: bool (default values is True)
    input for MINLP solver, indicates if factor shall be processed like int
m: float (default -.1/16)
    dy/dx, effective multiplier is a linear function f(x) = mx + n,
    m is the increase of that linear function
n: float (default 1.)
    effective multiplier is a linear function f(x) = mx + n, n is f(0)
step: int (default value -1)
    index of optimization step, for each step set to -1"""

def expand_def(mydef):
    """Creates afactor definitions for each step and id.

    Parameters
    ----------
    mydefk: Defk

    Returns
    -------
    iterator"""
    try:
        iter_steps = iter(mydef.step)
    except TypeError:
        iter_steps = iter([mydef.step])
    ids = mydef.id if isinstance(mydef.id, (list, tuple)) else [mydef.id]
    return (
        Factor(
            id_, mydef.type,
            (id_ if mydef.id_of_source is None else mydef.id_of_source),
            mydef.value, mydef.min, mydef.max, mydef.is_discrete,
            mydef.m, mydef.n, step)
        for id_, step in product(ids, iter_steps))

Terminallink = namedtuple(
    'Terminallink', 'branchid nodeid id step', defaults=(-1,))
Terminallink.__doc__ = """Links a branch terminal with a factor.

Parameters
----------
branchid: str
    ID of branch
nodeid: str
    ID of connectivity node
id: str
    unique identifier (for one step) of linked factor
step: int (default value -1)
    optimization step, defined for each step if -1"""

Injectionlink = namedtuple(
    'Injectionlink', 'injid part id step', defaults=(-1,))
Injectionlink.__doc__ = """Links an injection with a factor.

Parameters
----------
injid: str
    ID of injection
part: 'p'|'q'
    marker for active or reactive power to be multiplied
id: str
    unique identifier (for one step) of linked factor
step: int (default value -1)
    optimization step, defined for each step if -1"""

def injlink_(objid, id_, part, _, steps):
    """Creates instances of class cls.

    Parameters
    ----------
    objid: str, or list<str>, or tuple<str>
        id of object to link
    id_: str, or list<str>, or tuple<str>
        id of linked factor, accepts number of parts ids
        (one for 'p' or 'q', two for 'pq')
    part: 'p'|'q'|'pq'
        active power or reactive power
    _: str
        not used
    steps: int, or list<int>, or tuple<int>
        index of step"""
    try:
        iter_steps = iter(steps)
    except TypeError:
        iter_steps = iter([steps])
    objids = objid if isinstance(objid, (list, tuple)) else [objid]
    ids = id_ if isinstance(id_, (list, tuple)) else [id_]
    return [Injectionlink(objid_, t[0], t[1], step_)
            for step_, objid_, t in
                product(iter_steps, objids, zip(part, ids))]

def termlink_(id_of_branch, id_, nodeid, steps):
    """Creates instances of class cls.

    Parameters
    ----------
    id_of_branch: str, or list<str>, or tuple<str>
        id of object to link
    id_: str, or list<str>, or tuple<str>
        id of linked factor, accepts number of parts ids
        (one for 'p' or 'q', two for 'pq')
    nodeid: str, or list<str>, or tuple<str>
        ID of connectivity node
    steps: int, or list<int>, or tuple<int>
        index of step"""
    try:
        iter_steps = iter(steps)
    except TypeError:
        iter_steps = iter([steps])
    objids = (
        id_of_branch 
        if isinstance(id_of_branch, (list, tuple)) 
        else [id_of_branch])
    nodeids = nodeid if isinstance(nodeid, (list, tuple)) else [nodeid]
    factorids = id_ if isinstance(id_, (list, tuple)) else [id_]
    if (len(factorids)==1) and (1 < len(nodeids)):
        factorids *= len(nodeids)
    return [Terminallink(t[0], t[1], t[2], step_)
            for step_, t in
                product(iter_steps, zip(objids, nodeids, factorids))]

Klink = namedtuple(
    'Klink',
    'objid id part nodeid step',
    defaults=('pq', None, -1))
Klink.__doc__ = """Logical connection between injection and a factor.

Parameters
----------
objiid: str|iterable_of_str
    identifier of injection/branch
id: str|iterable_of_str
    identifier of scaling factor to connect, one identifier for each
    given value or argument 'part'
part: 'p'|'q'|iterable_of_two_char (default 'pq')
    identifies the attribute of the injection to multipy with factor
    ('p'/'q'- injected active/reactive power), the value is relevant
    in case argument 'cls' is Injectionlink only
nodeid: str
    ID of connectivity node, the value is relevant
    in case argument 'cls' is Terminallink only
step: int (default value -1)|iterable_of_int
    addresses the optimization step, first optimization step has index 0,
    defined for each step if -1"""

Tlink = namedtuple(
    'Tlink',
    'id_of_branch id_of_factor id_of_node step',
    defaults=('pq', None, -1))
Tlink.__doc__ = """Logical connection between a terminal of a branch
and a factor.

Parameters
----------
id_of_branch: str|iterable_of_str
    identifier of branch
id_of_factor: str|iterable_of_str
    identifier of scaling factor to connect, one identifier for each
    given value or argument 'part'
id_of_node: str
    ID of connectivity node, the value is relevant
    in case argument 'cls' is Terminallink only
step: int (default value -1)|iterable_of_int
    addresses the optimization step, first optimization step has index 0,
    defined for each step if -1"""

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

# class => (column_types, function_string_to_type, is_tuple?)
_attribute_types = {
     #    message level
     Message:(
         [object, np.int16],
         [_tostring, np.int16],
         [False, False]),
     #    id      type id_of_source value     min
     #    max     is_discrete   step
     Defk:(
         [object, object, object, np.float64, np.float64,
          np.float64, bool, np.float64, np.float64, np.int16],
         [_tostring, _tostring, _tostring, np.float64, np.float64,
          np.float64, bool, np.float64, np.float64, np.int16],
         [True, False, False, False, False, False, False, False, False, True]),
     #    id      type id_of_source value     min
     #    max     is_discrete   m    n   step
     Deft:(
         [object, object, object, np.float64, np.float64,
          np.float64, bool, np.float64, np.float64, np.int16],
         [_tostring, _tostring, _tostring, np.float64, np.float64,
          np.float64, bool, np.float64, np.float64, np.int16],
         [True, False, False, False, False, False, False, False, False, True]),
     #    id, type, id_of_source, value, min, max,
     #    is_discrete, m, n, step
     Factor:(
         [object, object, object, np.float64, np.float64, np.float64,
          bool, np.float64, np.float64, np.int16],
         [_tostring, _tostring, _tostring, np.float64, np.float64, np.float64,
          bool, np.float64, np.float64, np.int16],
         [False, False, False, False, False, False,
          False, False, False, False]),
     #    injid, part, id, step
     Injectionlink:(
         [object, object, object, np.int16],
         [_tostring,  _tostring, _tostring, np.int16],
         [False, False, False, False]),
     #    branchid, nodeid, id, step
     Terminallink:(
         [object, object, object, np.int16],
         [_tostring,  _tostring, _tostring, np.int16],
         [False, False, False, False]),
     #    objid   id      part    nodeid     step
     Klink:(
         [object, object, object, object, np.int16],
         [_tostring,  _tostring, _tostring, _tostring, np.int16],
         [True, True, True, True, True]),
     #    objid   id      part    nodeid     step
     Tlink:(
         [object, object, object, np.int16],
         [_tostring,  _tostring, _tostring, np.int16],
         [True, True, True, True]),
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
     #    id      id_of_node_A id_of_node_B y_lo   y_tr
     Branch:(
         [object, object, object, np.complex128, np.complex128],
         [_tostring, _tostring, _tostring, np.complex128, np.complex128],
         [False, False, False, False, False]),
     #    id_of_node V
     Slacknode:(
         [object, np.complex128],
         [_tostring, np.complex128],
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

def df_astype(df, cls_):
    """Casts types of pandas.DataFrame columns to types according
    to _attribute_types.

    Parameters
    ----------
    df: pandas.DataFrame
    cls_: class
        class of named tuple

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    Exception if data cannot be converted into required type"""
    return df.astype(dict(zip(cls_._fields, _attribute_types[cls_][0])))

def make_df(cls_, content=_EMPTY_TUPLE):
    """Creates a pandas.DataFrame instance with column types set according
    to _attribute_types.

    Parameters
    ----------
    cls_: class
        class of named tuple
    content: array_like (iterable?)
        instances of cls_

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    Exception when data cannot be converted into required type"""
    return df_astype(pd.DataFrame(content, columns=cls_._fields), cls_)

# frames with types for columns
SLACKNODES = make_df(Slacknode)
BRANCHES = make_df(Branch)
INJECTIONS = make_df(Injection)
OUTPUTS = make_df(Output)
PVALUES = make_df(PValue)
QVALUES = make_df(QValue)
IVALUES = make_df(IValue)
VVALUES = make_df(Vvalue)
FACTORS = make_df(Factor)
INJLINKS = make_df(Injectionlink)
TERMINALLINKS = make_df(Terminallink)
TERMS = make_df(Term)
MESSAGES = make_df(Message)
