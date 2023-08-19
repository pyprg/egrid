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
import re
from collections import namedtuple
from itertools import product

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
    defaults=('slack', 1.+.0j,))
Slacknode.__doc__ += ': Tag for a slack node'
Slacknode.id_of_node.__doc__ = (
    "str, optional, default 'slack' identifier of the slack node")
Slacknode.V.__doc__ = (
    "complex, optional, default 1+0j, voltage at slack node")

Injection = namedtuple(
    'Injection',
    'id id_of_node P10 Q10 Exp_v_p Exp_v_q',
    defaults=(0.0, 0.0, 0.0, 0.0))
Injection.__doc__ += (
    ": Model of an electrical one-terminal-device including "
    "consumers (positiv and negative loads), PQ- and PV-generators, "
    "batteries and shunt capacitors.")
Injection.id.__doc__ = (
    "str, unique identifier of injection")
Injection.id_of_node.__doc__ = (
    "str, id of connected node")
Injection.P10.__doc__ = (
    "float, (default value 0) "
    "active power when magnitude of voltage is 1.0 pu, "
    "the value is the sum for all three phases")
Injection.Q10.__doc__ = (
    "float, (default value 0) "
    "reactive power when magnitude of voltage is 1.0 pu "
    "the value is the sum for all three phases")
Injection.Exp_v_p.__doc__ = (
    "float, (default value 0) "
    "exponent for voltage dependency of active power, "
    "0.0 active power P is independent from voltage-magnitude e.g. generators "
    "2.0 for constant conductance")
Injection.Exp_v_q.__doc__ = (
    "float, (default value 0) "
    "exponent for voltage dependency of active power, "
    "0.0 reactive power is independent from voltage-magnitude e.g. generators"
    "2.0 for constant susceptance")

Output = namedtuple(
    'Output',
    'id_of_batch id_of_device id_of_node', defaults=(None,))
Output.__doc__ += (
    ": Measured terminal or terminal of a device which is part "
    "of a group of devices whose flow is measured.")
Output.id_of_batch.__doc__ = (
    "str, unique identifier of the batch")
Output.id_of_device.__doc__ = (
    "str, id referencing a branch or an injection")
Output.id_of_node.__doc__ = (
    "str, (default value None) "
    "id of the connected node, 'None' if at injection")

PValue = namedtuple(
    'PValue',
    'id_of_batch P direction cost',
    defaults=(0., 1., 0.))
PValue.__doc__ += """: Value of (measured) active power.

The optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Output instance(s) having the identical 'id_of_batch' value must
exist. Placement of multiple measurements in switch fields combined with
several branches or injections are modeled by PValue and Output instances
sharing the same 'id_of_batch'-value.

Attribute cost is introduced for Volt-Var-Control."""
PValue.id_of_batch.__doc__ = (
    "str, unique identifier of the point")
PValue.P.__doc__ = (
    "float, active power, sum of all three phases")
PValue.direction.__doc__ = (
    "float (default value 1) -1 or 1")
PValue.cost.__doc__ = (
    "float (default value 0) cost")

QValue = namedtuple(
    'QValue',
    'id_of_batch Q direction cost',
    defaults=(0., 1., 0.))
QValue.__doc__ += """: Value of (measured) reactive power.

The optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Output instance(s) having the identical 'id_of_batch' value
must exist. Placement of multiple measurements in switch fields combined with
several branches or injections are modeled by QValue and Output instances
sharing the same 'id_of_batch'-value.

Attribute cost is introduced for Volt-Var-Control."""
QValue.id_of_batch.__doc__ = (
    "str, unique identifier of the point")
QValue.Q.__doc__ = (
    "float, reactive power, sum of all three phases")
QValue.direction.__doc__ = (
    "float, (default value 1) -1 or 1")
QValue.cost.__doc__ = (
    "float, (default value 0) cost")

IValue = namedtuple(
    'IValue',
    'id_of_batch I', defaults=(0.0,))
IValue.__doc__ += """: Value of (measured) electric current. The
optimization (estimation) target is to meet those (and other given) values.
When the measurement is placed at a terminal of a branch or injection a
corresponding Output instance having the identical 'id_of_batch' value
must exist. Placement of multiple measurements in switch fields combined with
several branches or injections are modeled by IValue and Output instances
sharing the same 'id_of_batch'-value."""
IValue.id_of_batch.__doc__ = (
    "str, unique identifier of the point")
IValue.I.__doc__ = (
    "float, magnitude of electric current, value for one phase")

Vvalue = namedtuple(
    'Vvalue',
    'id_of_node V', defaults=(1.,))
Vvalue.__doc__ += """: Values of (measured) electric voltage. The
optimization (estimation) target is to meet those (and other given) values."""
Vvalue.id_of_node.__doc__ = (
    "str, unique identifier of node the voltage was measured at or the "
    "setpoint is for")
Vvalue.V.__doc__ = (
    "float, (default value 1.0) magnitude of electric voltage")

Vlimit = namedtuple(
    'Vlimit',
    'id_of_node min max step', defaults=("", 0.9, 1.1, -1))
Vlimit.__doc__ += ": Limits of node voltage."
Vlimit.id_of_node.__doc__ = (
    "str, optional, empty string "
    "identifier of connectivity node, empty string addresses all nodes")
Vlimit.min.__doc__ = (
    "float, optional, default 0.9 "
    "smallest possible magnitude of the voltage at node")
Vlimit.max.__doc__ = (
    "float, optional, default 1.1 "
    "greatest possible magnitude of the voltage at node")
Vlimit.step.__doc__ = (
    "int, optional, default -1 "
    "index of optimization step, -1 for all steps")

DEFAULT_FACTOR_ID = '_default_'

Factor = namedtuple(
    'Factor',
    'id type id_of_source value min max is_discrete m n step cost',
    defaults=(
        'var', DEFAULT_FACTOR_ID, 1.0, -np.inf, np.inf, False, 1., 0., -1, 1.))
Factor.__doc__ += ": Data of a factor."
Factor.id.__doc__ = (
    "str, unique idendifier of factor")
Factor.type.__doc__ = (
    "'var'|'const' (default value 'var'), "
    "decision variable or parameter")
Factor.id_of_source.__doc__ = (
    "str (default value DEFAULT_FACTOR_ID) "
    "id of scaling factor (previous optimization step) for initialization, "
    "default value ID of default factor")
Factor.value.__doc__ = (
    "float (default vaLue 1), "
    "used by initialization if no source factor in previous optimization step")
Factor.min.__doc__ = (
    "float (default value -numpy.inf), smallest possible value")
Factor.max.__doc__ = (
    "float, (default value numpy.inf), greatest possible value")
Factor.is_discrete.__doc__ = (
    "bool (default is False), "
    "no values after decimal point if True, input for solver accepted "
    "by MINLP solvers")
Factor.m.__doc__ = (
    "float (default 1.), "
    "dy/dx, effective multiplier is a linear function f(x) = mx + n, m is the "
    "increase of that linear function, value is applied for tap factors only")
Factor.n.__doc__ = (
    "float (default 0.), "
    "effective multiplier is a linear function f(x) = mx + n, n is f(0), "
    "value is applied for tap factors only")
Factor.step.__doc__ = (
    "int (default value -1), "
    "index of optimization step, defined for each step if set to -1")
Factor.cost.__doc__ = (
    "float (default 1.), cost of change (for Volt-Var-Control)")

Terminallink = namedtuple(
    'Terminallink', 'branchid nodeid id step', defaults=(-1,))
Terminallink.__doc__ += ": Links a branch terminal with a factor."
Terminallink.branchid.__doc__ = (
    "str, ID of branch")
Terminallink.nodeid.__doc__ = (
    "str, ID of connectivity node")
Terminallink.id.__doc__ = (
    "str, unique identifier (for one step) of linked factor")
Terminallink.step.__doc__ = (
    "int (default value -1) "
    "optimization step, defined for each step if -1")

Injectionlink = namedtuple(
    'Injectionlink', 'injid part id step', defaults=(-1,))
Injectionlink.__doc__ += ": Links an injection with a factor."
Injectionlink.injid.__doc__ = (
    "str, ID of injection")
Injectionlink.part.__doc__ = (
    "'p'|'q', marker for active or reactive power to be multiplied")
Injectionlink.id.__doc__ = (
    "str, unique identifier (for one step) of linked factor")
Injectionlink.step.__doc__ = (
    "int (default value -1) "
    "optimization step, defined for each step if -1")

Klink = namedtuple(
    'Klink',
    'id_of_injection part id_of_factor step',
    defaults=(-1,))
Klink.__doc__ += ": Logical connection between injection and a factor."
Klink.id_of_injection.__doc__ = (
    "str|iterable_of_str, identifier of injection")
Klink.part.__doc__ = (
    "'p'|'q'|iterable_of_p_q "
    "identifies the attribute of the injection to multipy with "
    "('p'/'q'- injected active/reactive power)")
Klink.id_of_factor.__doc__ = (
    "str|iterable_of_str, "
    "identifier of scaling factor to connect to, one identifier for each "
    "given value of argument 'part'")
Klink.step.__doc__ = (
    "int|iterable_of_int (default value -1), "
    "addresses the optimization step, first optimization step has index 0, "
    "defined for each step if -1")

Tlink = namedtuple(
    'Tlink',
    'id_of_node id_of_branch id_of_factor step',
    defaults=(-1,))
Tlink.__doc__ += (
    ": Logical connection between a terminal of a branch and a factor.")
Tlink.id_of_node.__doc__ = (
    "str|iterable_of_str, ID of connectivity node")
Tlink.id_of_branch.__doc__ = (
    "str|iterable_of_str, identifier of branch")
Tlink.id_of_factor.__doc__ = (
    "str|iterable_of_str, "
    "identifier of taps (terminal) factor to connect, "
    "id_of_node, id_of_branch, id_of_factor shall have the same lenght "
    "if iterable")
Tlink.step.__doc__ = (
    "int (default value -1)|iterable_of_int "
    "addresses the optimization step, first optimization step has index 0, "
    "defined for each step if -1")

Term = namedtuple(
    'Term',
    'id args fn weight step',
    defaults=([''], 'diff', 1., -1))
Term.__doc__ += ": Data of an ojective-function-term."
Term.id.__doc__ = (
    "str, unique identifier of a term")
Term.args.__doc__ = (
    "list, str, references to an arguments of function fn")
Term.fn.__doc__ = (
    "str, indentifier of function for calculating a term of the "
    "objective function")
Term.weight.__doc__ = (
    "float, multiplier for term in objective function")
Term.step.__doc__ = (
    "int, index of optimization step")

Defoterm = namedtuple(
    'Defoterm',
    'fn args weight step',
    defaults=('diff', '', 1., -1))
Defoterm.__doc__ += """: Definition for a mathematical term of the objective
function."""
Defoterm.fn.__doc__ = (
    "str, optional, default 'diff', name of function")
Defoterm.args.__doc__ = (
    "str | iterable_of_str, optional, default '' "
    "name of argument or name of arguments")
Defoterm.weight.__doc__ = (
    "float, optional, default 1.0 multiplier for term in objective function")
Defoterm.step.__doc__ = (
    "int | iterable_of_int, optional, default -1 "
    "optimization step, -1 means all steps")

def expand_defoterm(index,  defoterm):
    """Creates definitions of terms for objective function (Term).

    Parameters
    ----------
    index: int
        index of defoterm
    defoterm: Defoterm

    Returns
    -------
    iterator
        Term"""
    try:
        iter_steps = iter(defoterm.step)
    except TypeError:
        iter_steps = iter([defoterm.step])
    args = (
        defoterm.args
        if isinstance(defoterm.args, (list, tuple)) else [defoterm.args])
    return (
        Term(
            id=str(index),
            args=args,
            weight=defoterm.weight,
            fn=defoterm.fn,
            step=step)
        for step in iter_steps)

Message = namedtuple('Message', 'message level', defaults=('', 2))
Message.__doc__ += ": Information, warning or error message."
Message.message.__doc__ = (
    "str, human readable text")
Message.level.__doc__ = (
    "int, 0 - information, 1 - warning, 2 - error")

# helper

def expand_def(mydef):
    """Creates factor definitions for each step and id.

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
            mydef.m, mydef.n, step, mydef.cost)
        for id_, step in product(ids, iter_steps))

# convenience

Defk = namedtuple(
    'Defk',
    'id type id_of_source value min max is_discrete m n step cost',
    defaults=('var', None, 1.0, -np.inf, np.inf, False, 1., 0., -1, 0.))
Defk.__doc__ += ": Definition of a scaling factor."
Defk.id.__doc__ = (
    "str|iterable_of_str, "
    "identifier of scaling factor, unique among factors of same step")
Defk.type.__doc__ = (
    "'var'|'const' (default value 'var'), "
    "'var' - factor is a decision variable "
    "'const' - factor is a parameter")
Defk.id_of_source.__doc__ = (
    "str (default value None), "
    "identifies factor of previous estimation step whose value "
    "will be used for initialization")
Defk.value.__doc__ = (
    "float (default value 1), "
    "used for initialization if 'id_of_source' does not reference a "
    "scaling factor (of previous optimization step)")
Defk.min.__doc__ = (
    "float (default value -numpy.inf), "
    "smallest value allowed")
Defk.max.__doc__ = (
    "float (default value numpy.inf), "
    "greatest value allowed")
Defk.is_discrete.__doc__ = (
    "bool (default values is False), "
    "input for MINLP solver, indicates if factor shall be processed like int")
Defk.m.__doc__ = (
    "float (default 1.), "
    "dy/dx, effective multiplier is a linear function f(x) = mx + n, "
    "m is the increase of that linear function "
    "not used")
Defk.n.__doc__ = (
    "float (default 0.), "
    "effective multiplier is a linear function f(x) = mx + n, n is f(0) "
    "not used")
Defk.step.__doc__ = (
    "int | iterable_of_int (default value -1), "
    "index of optimization step, for each step set to -1")
Defk.cost.__doc__ = (
    "float (default 0.), cost of change (for Volt-Var-Control)")

Deft = namedtuple(
    'Deft',
    'id type id_of_source value min max is_discrete m n step cost',
    defaults=('const', None, 0., -16., 16., True, -.1/16, 1., -1, 0.))
Deft.__doc__ += ": Definition of a taps (terminal) factor."
Deft.id.__doc__ = (
    " str|iterable_of_str, "
    "identifier of scaling factor, unique among factors of same step")
Deft.type.__doc__ = (
    "'var'|'const' (default value 'const'), "
    "'var' - factor is a decision variable "
    "'const' - factor is a parameter")
Deft.id_of_source.__doc__ = (
    "str (default value None), "
    "identifies factor of previous estimation step whose value "
    "will be used for initialization")
Deft.value.__doc__ = (
    "float (default value 0), "
    "used for initialization if 'id_of_source' does not reference a "
    "scaling factor (of previous optimization step)")
Deft.min.__doc__ = (
    "float (default value -16), smallest value allowed")
Deft.max.__doc__ = (
    "float (default 16), greatest value allowed")
Deft.is_discrete.__doc__ = (
    "bool (default values is True); "
    "input for MINLP solver, indicates if factor shall be processed like int")
Deft.m.__doc__ = (
    "float (default -.1/16), "
    "dy/dx, effective multiplier is a linear function f(x) = mx + n, "
    "m is the increase of that linear function")
Deft.n.__doc__ = (
    "float (default 1.), "
    "effective multiplier is a linear function f(x) = mx + n, n is f(0)")
Deft.step.__doc__ = (
    "int | iterable_of_int (default value -1), "
    "index of optimization step, for each step set to -1")
Deft.cost.__doc__ = (
    "float (default 0.), "
    "cost of change (for Volt-Var-Control)")

def _iterable(item):
    return item if isinstance(item, (list, tuple)) else[item]

def expand_klink(id_of_injection, part, id_of_factor, steps):
    """Creates instances of class Injectionlink.

    Parameters
    ----------
    id_of_injection: str, or list<str>, or tuple<str>
        id of object to link
    part: 'p'|'q'|'pq'
        active power or reactive power
    id_of_factor: str, or list<str>, or tuple<str>
        id of linked factor, accepts number of parts ids
        (one for 'p' or 'q', two for 'pq')
    steps: int, or list<int>, or tuple<int>
        index of step"""
    try:
        iter_steps = iter(steps)
    except TypeError:
        iter_steps = iter([steps])
    objids = _iterable(id_of_injection)
    ids = _iterable(id_of_factor)
    return [
        Injectionlink(objid_, t[0], t[1], step_)
        for step_, objid_, t in product(iter_steps, objids, zip(part, ids))]

def expand_tlink(id_of_node, id_of_branch, id_of_factor, steps):
    """Creates instances of class Terminallink.

    Parameters
    ----------
    id_of_node: str, or list<str>, or tuple<str>
        ID of connectivity node
    id_of_branch: str, or list<str>, or tuple<str>
        id of object to link
    id_of_factor: str, or list<str>, or tuple<str>
        id of linked factor, accepts number of parts ids
        (one for 'p' or 'q', two for 'pq')
    steps: int, or list<int>, or tuple<int>
        index of step"""
    try:
        iter_steps = iter(steps)
    except TypeError:
        iter_steps = iter([steps])
    objids = _iterable(id_of_branch)
    nodeids = _iterable(id_of_node)
    factorids = _iterable(id_of_factor)
    if (len(factorids)==1) and (1 < len(nodeids)):
        factorids *= len(nodeids)
    return [
        Terminallink(t[0], t[1], t[2], step_)
        for step_, t in product(iter_steps, zip(objids, nodeids, factorids))]

Defvl = namedtuple(
    'Defvl',
    'id_of_node min max step', defaults=('', 0.9, 1.1, -1))
Defvl.__doc__ = ": Define limit of node voltage."
Defvl.id_of_node.__doc__ = (
    "str|iterable_of_str, optional, default '' "
    "identifier of connectivity node(s), addresses all nodes if empty string")
Defvl.min.__doc__ = (
    "float, optional, default 0.9, "
    "smallest possible magnitude of the voltage at node")
Defvl.max.__doc__ = (
    "float, optional, default 1.1, "
    "greatest possible magnitude of the voltage at node")
Defvl.step.__doc__ = (
    "int|iterable_of_int, optional, default -1, "
    "index of optimization step, -1 for all steps")

def expand_defvl(defvl):
    """Creates instances of Vlimit.

    Parameters
    ----------
    defvl : defvl
        definition of voltage limits for nodes and steps

    Returns
    -------
    list
        Vlimit"""
    nodeids = _iterable(defvl.id_of_node)
    steps = _iterable(defvl.step)
    return [
        Vlimit(nodeid_, defvl.min, defvl.max, step_)
        for nodeid_, step_ in product(nodeids, steps)]

# meta data for use by graphparser, and for creation of empty pandas.DataFrame
#   instances with correct column types

def _tostring(string):
    return string[1:-1] if string.startswith(('\'', '"')) else string

def _tobool(string):
    return False if string.upper() == 'FALSE' else bool(string)

def _tofloat(string):
    return np.float64(e3(string))

# class => (column_types, function_string_to_type, is_tuple?)
_attribute_types = {
     #    message level
     Message:(
         [object, np.int16],
         [_tostring, np.int16],
         [False, False]),
     #    id      type id_of_source value     min
     #    max     is_discrete   step       cost
     Defk:(
         [object, object, object, np.float64, np.float64,
          np.float64, bool, np.float64, np.float64, np.int16, np.float64],
         [_tostring, _tostring, _tostring, _tofloat, _tofloat,
          _tofloat, _tobool, _tofloat, _tofloat, np.int16, _tofloat],
         [True, False, False, False, False, False, False, False, False, True,
          False]),
     #    id      type id_of_source value     min
     #    max     is_discrete   m    n   step
     Deft:(
         [object, object, object, np.float64, np.float64,
          np.float64, bool, np.float64, np.float64, np.int16, np.float64],
         [_tostring, _tostring, _tostring, _tofloat, _tofloat,
          _tofloat, _tobool, _tofloat, _tofloat, np.int16, _tofloat],
         [True, False, False, False, False, False, False, False, False, True,
          False]),
     #    id_of_node min max step
     Defvl:([object, np.float64, np.float64, np.int16],
            [_tostring, np.float64, np.float64, np.int16],
            [True, False, False, True]),
     #    fn args step
     Defoterm:(
         [object, object, np.float64, np.int32],
         [_tostring, _tostring, _tofloat, np.int32],
         [False, True, False, True]),
     #    id, type, id_of_source, value, min, max,
     #    is_discrete, m, n, step, cost
     Factor:(
         [object, object, object, np.float64, np.float64, np.float64,
          bool, np.float64, np.float64, np.int16, np.float64],
         [_tostring, _tostring, _tostring, _tofloat, _tofloat, _tofloat,
          _tobool, _tofloat, _tofloat, np.int16, _tofloat],
         [False, False, False, False, False, False,
          False, False, False, False, False]),
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
     #    id       arg     fn  weight   step
     Term:(
         [object,  object, object, np.float64, np.int32],
         [_tostring, _tostring, _tostring, _tofloat, np.int32],
         [False, False, False, False, False]),
     #    id_of_batch id_of_device id_of_node
     Output:(
         [object, object, object],
         [_tostring, _tostring, _tostring],
         [False, False, False]),
     #    id_of_batch   P     direction
     PValue:(
         [object, np.float64, np.float64, np.float64],
         [_tostring,  _tofloat, _tofloat, _tofloat],
         [False, False, False, False]),
     #    id_of_batch   Q     direction
     QValue:(
         [object, np.float64, np.float64, np.float64],
         [_tostring, _tofloat, _tofloat, _tofloat],
         [False, False, False, False]),
     #    id_of_batch   I
     IValue:(
         [object, np.float64],
         [_tostring, _tofloat],
         [False, False]),
     #    id_of_node  V
     Vvalue:(
         [object, np.float64],
         [_tostring, _tofloat],
         [False, False]),
     #     id_of_node min max step
     Vlimit:(
         [object, np.float64, np.float64, np.int16],
         [_tostring, _tofloat, _tofloat, np.int16],
         [False, False, False, False]),
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
         [_tostring, _tostring, _tofloat, _tofloat, _tofloat, _tofloat],
         [False, False, False, False, False, False])}

meta_of_types = [
    (cls_, tuple(zip(*info[1:3]))) for cls_, info in _attribute_types.items()]
_EMPTY_TUPLE = ()

def df_astype(df, cls_):
    """Casts types of pandas.DataFrame columns to _attribute_types.

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
    """Creates a pandas.DataFrame instance.

    Types of columns are according to _attribute_types.

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

# frames with correct types for columns
SLACKNODES = make_df(Slacknode)
BRANCHES = make_df(Branch)
INJECTIONS = make_df(Injection)
OUTPUTS = make_df(Output)
PVALUES = make_df(PValue)
QVALUES = make_df(QValue)
IVALUES = make_df(IValue)
VVALUES = make_df(Vvalue)
VLIMITS = make_df(Vlimit)
FACTORS = make_df(Factor)
INJLINKS = make_df(Injectionlink)
TERMINALLINKS = make_df(Terminallink)
TERMS = make_df(Term)
MESSAGES = make_df(Message)
