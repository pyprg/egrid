# egrid

## Purpose

Prepocessed model of a balanced electric distribution network for experimental
power flow calculation and state estimation. Instances of Model can be the
input for (a variety of) power flow calculation or estimation algorithms. They
provide an easy to use structure for calculating current and power flow
through lines and into consumers (using an additional voltage vector which
is the result of a power-flow-calculation).

Function **make_model(\*args)** creates an instance of Model from  arguments
of type

    - Slacknode
    - Branch (line, series capacitor, transformer winding, transformer,
      closed switch)
    - Injection (consumer, shunt capacitor, PQ/PV-generator, battery)
    - Output (indicates that measured flow (I, P or Q) or a part thereof
      flows through the referenced terminal (device or node+device))
    - PValue (measured active power)
    - QValue (measured reactive power)
    - IValue (measured electric current)
    - Vvalue (measured voltage or setpoint)
    - Defk/Deft (definition of a scaling/termional factor, for estimation)
    - Defvl (definition of voltage limits)
    - Klink/Tlink (associates a factor to an injection or terminal of a branch)

including tuples, lists and iterables thereof (for a power-flow-calculation
just Slacknode ... Branchtaps are necessary).
Additionally, __make_model__ can consume network descriptions as multiline
strings if package 'graphparser' is installed. This method is intended to
input very small electric networks using a text editor.

Most fields of Model instances provide pandas.DataFrame instances. Electric
values are stored per unit. Branch models are PI-equivalent circuits. Active
and reactive power of injections have a dedicated voltage exponent.

## Details of egrid.model.Model

Fields of egrid.model.Model
---------------------------
nodes: pandas.DataFrame (id of node)

    * .idx, int, index of power-flow-calculation node

slacks: pandas.DataFrame

    * .id_of_node, str, id of connection node
    * .V, complex, given voltage at this slack
    * .index_of_node, int, index of power-flow-calculation node

injections: pandas.DataFrame

    * .id, str, unique identifier of injection
    * .id_of_node, str, unique identifier of connected node
    * .P10, float, active power at voltage magnitude 1.0 pu
    * .Q10, float, reactive power at voltage magnitude 1.0 pu
    * .Exp_v_p, float, voltage dependency exponent of active power
    * .Exp_v_q, float, voltage dependency exponent of reactive power
    * .index_of_node, int, index of connected power-flow-calculation node

terminal_to_branch: numpy.array

    * [0, br] indices of terminal A
    * [1, br] indices of terminal B
    index of branch is the column index

branchterminals: pandas.DataFrame

    * .index_of_branch, int, index of branch
    * .id_of_branch, str, unique idendifier of branch
    * .id_of_node, str, unique identifier of connected node
    * .id_of_other_node, str, unique identifier of node connected
       at other side of the branch
    * .index_of_node, int, index of connected power-flow-calculation node
    * .index_of_other_node, int, index of power-flow-calculation node connected
       at other side of the branch
    * .y_lo, complex, longitudinal branch admittance
    * .y_tr_half, complex, half of transversal branch admittance
    * .g_lo, float, longitudinal conductance
    * .b_lo, float, longitudinal susceptance
    * .g_tr_half, float, transversal conductance of branch devided by 2
    * .b_tr_half, float, transversal susceptance of branch devided by 2
    * .side, str, 'A' | 'B', side of branch, first or second

bridgeterminals: pandas.DataFrame

    * .index_of_branch, int, index of branch
    * .id_of_branch, str, unique idendifier of branch
    * .id_of_node, str, unique identifier of connected node
    * .id_of_other_node, str, unique identifier of node connected
       at other side of the branch
    * .index_of_node, int, index of connected power-flow-calculation node
    * .index_of_other_node, int, index of power-flow-calculation node connected
       at other side of the branch
    * .y_lo, complex, longitudinal branch admittance
    * .y_tr_half, complex, half of transversal branch admittance
    * .g_lo, float, longitudinal conductance
    * .b_lo, float, longitudinal susceptance
    * .g_tr_half, float, transversal conductance of branch devided by 2
    * .b_tr_half, float, transversal susceptance of branch devided by 2
    * .side, str, 'A' | 'B', side of branch, first or second

branchoutputs: pandas.DataFrame

    * .id_of_batch, str, unique identifier of measurement batch
    * .id_of_node, str, id of node connected to branch terminal
    * .id_of_branch, str, unique identifier of branch
    * .index_of_node, int, index of power-flow-calculation node connected
       to branch terminal
    * .index_of_branch, int, index of branch

injectionoutputs: pandas.DataFrame

    * .id_of_batch, str, unique identifier of measurement batch
    * .id_of_injection, str, unique identifier of injection
    * .index_of_injection, str, index of injection

pvalues: pandas.DataFrame

    * .id_of_batch, str, unique identifier of measurement batch
    * .P, float, active power
    * .direction, float, -1: from device into node, 1: from node into device

qvalues: pandas.DataFrame

    * .id_of_batch, str, unique identifier of measurement batch
    * .Q, float, reactive power
    * .direction, float, -1: from device into node, 1: from node into device

ivalues: pandas.DataFrame

    * .id_of_batch, str, unique identifier of measurement batch
    * .I, float, electric current

vvalues: pandas.DataFrame

    * .id_of_node, str, unique identifier of node voltage is given for
    * .V, float, magnitude of voltage
    * .index_of_node, index of node voltage is given for

shape_of_Y: tuple (int, int)

    shape of admittance matrix for power flow calculation

count_of_slacks: int

    count_of_slacks

factors: egrid.factors.Factors (namedtuple)

    * .gen_factordata, pandas.DataFrame (index: ('step','id'))
        * .step, -1
        * .type, 'var'|'const', type of factor decision variable or parameter
        * .id_of_source, str, id of factor (previous optimization step)
           for initialization
        * .value, float, used by initialization if no source factor in previous
           optimization step
        * .min, float
           smallest possible value
        * .max, float
           greatest possible value
        * .is_discrete, bool
           just 0 digits after decimal point if True, input for solver,
           accepted by MINLP solvers
        * .m, float,
           multiplier,
           the effective value of the factor is a linear function
           of var/const (mx + n)
        * .n, float,
           value of factor when var/const is 0,
           the effective value of the factor is a linear function
           of var/const (mx + n)
        * .index_of_symbol, int
        * .cost, float, cost of changing (for Volt-Var-control)

    * .gen_injfactor, pandas.DataFrame (index: ('id_of_injection', 'part'))

        * .step, -1 (int)
        * id, str, ID of factor

    * .terminalfactors, pandas.DataFrame

        * .id, str, identifier of factor
        * .index_of_terminal, int
        * .index_of_other_terminal, int
        * .type, 'var'|'const'
        * .id_of_source, str
        * .value, float
        * .min, float
        * .max, float
        * .is_discrete, bool
        * .m, float
        * .n, float
        * .index_of_symbol, int
        * .cost, float, cost of changing (for Volt-Var-control)

    * .get_groups: function (iterable_of_int) -> (pandas.DataFrame)

        pandas.DataFrame(index: ('step', 'id'))

            * .type, 'var'|'const', type of factor decision
               variable or parameter
            * .id_of_source, str, id of factor (previous optimization step)
               for initialization
            * .value, float, used by initialization if no source factor
               in previous optimization step
            * .min, float
               smallest possible value
            * .max, float
               greatest possible value
            * .is_discrete, bool
               just 0 digits after decimal point if True, input for solver,
               accepted by MINLP solvers
            * .m, float
               increase of multiplier with respect to change of var/const
               the effective multiplier is a linear
               function of var/const (mx + n)
            * .n, float
               multiplier when var/const is 0.
               the effective multiplier is a linear function of
               var/const (mx + n)
            * .cost, float, cost of changing (for Volt-Var-control)

    * .get_injfactorgroups: function (iterable_of_int) -> (pandas.DataFrame)

        pandas.DataFrame (index: ('step', 'id_of_injection', 'part'))

            * .id, str, ID of factor

mnodeinj: scipy.sparse.csc_matrix

    converts a vector ordered according to injection indices to a vector
    ordered according to power flow calculation nodes (adding values of
    injections for each node) by calculating 'mnodeinj @ vector'

messages: pandas.DataFrame

    * .message, str, message on reason of error
    * .level, int, 0 - information, 1 - warning. 2 - error

## Making a Model

Function **model_from_frames** consumes a dictionary of
pandas.DataFrames. **model_from_frames** aggregates nodes connected without
impedance, creates indices, arranges data per branch-terminal from branch-data,
calculates values of branches from admittances.

Function make_model generates a model from network device objects defined
in **egrid.builder**.


Example - 3 nodes, 2 lines, 1 consumer:
```
node: 0               1               2

      |      line     |     line      |
      +-----=====-----+-----=====-----+
      |               |               |
                                     \|/ consumer
```

Python code for example, suitable input for function **egrid.make_model**
(Branchtap is for demo only, it is used with transformers,
however, transformers/transformerwindings are modeled using class Branch too.):
```
from egrid.builder import (
    Slacknode, PValue, QValue, IValue, Output, Branch,
    Injection, Defk, Deft, Klink, Tlink)

example = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    PValue(
        id_of_batch='pq_line_0',
        P=30.),
    QValue(
        id_of_batch='pq_line_0',
        Q=8.),
    Output(
        id_of_batch='pq_line_0',
        id_of_node='n_0',
        id_of_device='line_0'),
    IValue(
        id_of_batch='i_line_0',
        I=40.0),
    Output(
        id_of_batch='i_line_0',
        id_of_node='n_0',
        id_of_device='line_0'),
    Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    Branchtaps(
        id='taps_0',
        id_of_node='n_0',
        id_of_branch='line_0',
        Vstep=.1/16,
        positionmin=-16,
        positionneutral=0,
        positionmax=16,
        position=0),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    Output(
        id_of_batch='pq_consumer_0',
        id_of_device='consumer_0'),
    Output(
        id_of_batch='i_consumer_0',
        id_of_device='consumer_0'),
    Injection(
        id='consumer_0',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=2.0,
        Exp_v_q=2.0),
    Defk(step=(0, 1, 2), id=('kp', 'kq')),
    Klink(step=(0, 1, 2), objid='consumer_0', part='pq', id=('kp', 'kq'))]
```

Valid input to **make_model** is a multiline pseudo graphic string e.g.
this one:
```
               y_tr=1e-6+1e-6j                 y_tr=1e-6+1e-6j
slack=True     y_lo=1e3-1e3j                   y_lo=1e3-1e3j
n0(---------- line_0 ----------)n1(---------- line_1 ----------)n2
                                |                               |
                                n1->> load0_1_        _load1 <<-n2->> load1_1_
                                |      P10=30.0         P10=20.7       P10=4.3
                                |      Q10=5            Q10=5.7        Q10=2
                                |
                                |              y_lo=1e3-1e3j
                                |              y_tr=1e-6+1e-6j
                                n1(---------- line_2 ----------)n3
                                                                |
                                                      _load2 <<-n3->> load2_1_
                                                        P10=20.7       P10=20
                                                        Q10=5.7        Q10=5.7
```
