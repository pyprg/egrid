# egrid

## Purpose

Model of a balanced electric distribution network for experimental power flow 
calculation and state estimation. Model shall provide the data in a structure
supporting e.g. admittance matrix creation and result processing. Model is a 
namedtuple. Most of the fields provide pandas.DataFrames. Electric values are 
stored per unit.
The main function is make_model(*args). The arguments are of type 

    - Branch (line, series capacitor, transformer winding, transformer, closed switch)
    - Slacknode
    - Injection (consumer, shunt capacitor, PQ/PV-generator, battery)
    - Output (indicates that measured flow (I or PQ) or a part thereof flows through the referenced terminal (device or node+device))
    - PValue (measured active power)
    - QValue (measured reactive power)
    - PQValue (measured active and reactive power)
    - IValue (measured electric current)
    - Vvalue (measured voltage or setpoint)
    - Branchtaps
    - Defk (definition of a scaling factor, for estimation)
    - Link (associates a scaling factor to a load)
    
including tuples, lists and iterables thereof (for a power-flow-calculation
just Branch ... Branchtaps are necessary).
Additionally, __make_model__ can consume network descriptions as multiline 
strings if package 'graphparser' is installed. This method is intended to
input very small electric networks using a text editor. However, this method
is suitable for power flow calculation and must be extended with 
additional objects if taps and measurements/setpoints shall be included
for further processing.
    
Branch models are PI-equivalent circuits. Active and reactive power of
injections have a dedicated voltage exponent.

## Details of egrid.model.Model

Fields of egrid.model.Model
---------------------------
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
    * .y_mn, complex, longitudinal branch admittance
    * .y_mm_half, complex, half of transversal branch admittance
    * .y_tot, complex, y_mn + y_mm_half
    * .g_tot, float, conductance, g_mn + g_mm_half
    * .b_tot, float, susceptande, b_mn + b_mm_half
    * .g_mn, float, longitudinal conductance
    * .b_mn, float, longitudinal susceptance
    * .g_mm_half, float, transversal conductance of branch devided by 2
    * .b_mm_half, float, transversal susceptance of branch devided by 2
    * .side, str, 'A' | 'B', side of branch, first or second
    
branchoutputs: pandas.DataFrame

    * .id_of_batch, str, unique identifier of measurement batch
    * .id_of_node, str, id of node connected to branch terminal
    * .id_of_branch, str, unique identifier of branch
    * .index_of_node, int, index of node connected to branch terminal
    * .index_of_branch, int, index of branch
    
injectionoutputs: pandas.DataFrame

    * .id_of_batch, str, unique identifier of measurement batch 
    * .id_of_injection, str, unique identifier of injection
    * .index_of_injection, str, index of injection
    
pvalues: pandas.DataFrame

    * .id_of_batch, unique identifier of measurement batch
    * .P, float, active power
    * .direction, float, -1: from device into node, 1: from node into device
    
qvalues: pandas.DataFrame

    * .id_of_batch, unique identifier of measurement batch
    * .Q, float, reactive power
    * .direction, float, -1: from device into node, 1: from node into device
    
pqvalues: pandas.DataFrame

    * .id_of_batch, unique identifier of measurement batch
    * .P, float, active power
    * .Q, float, reactive power
    * .direction, float, -1: from device into node, 1: from node into device
    
ivalues: pandas.DataFrame

    * .id_of_batch, unique identifier of measurement batch
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

    True if index is index of slack node, false otherwise

load_scaling_factors: pandas.DataFrame

injection_factor_associations: pandas.DataFrame

messages: pandas.DataFrame

    * .errormessage, str, message on reason of error
    
## Making a Model

Function **model.model_from_frames** consumes an dictionary of 
pandas.DataFrames. **model_from_frames** creates indices, arranges data
per branch-terminal from branch-data, calculates values of branches from
admittances. The function is useful if data are given in a similar structure.

Function make_model generates a model from network device objects defined
in **egrid.builder** (see paragraph 'Purpose').


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
    Slacknode, PQValue, IValue, Output, Branch, Branchtaps, Injection,
    Defk, Link)

example = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    PQValue(
        id_of_batch='pq_line_0', 
        P=30., 
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
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Branchtaps(
        id='taps_0',
        id_of_node='n_0', 
        id_of_branch='line_0', 
        Vstep=.2/33, 
        positionmin=-16, 
        positionneutral=0, 
        positionmax=16,
        position=0),
    Branch(
        id='line_1',
        id_of_node_A='n_1', 
        id_of_node_B='n_2',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
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
    Link(step=(0, 1, 2), objid='consumer_0', part='pq', id=('kp', 'kq'))]
```

Valid input to **make_model** is a multiline pseudo graphic string e.g. 
this one:
```
               y_mm_half=1e-6+1e-6j            y_mm_half=1e-6+1e-6j
slack=True     y_mn=1e3-1e3j                   y_mn=1e3-1e3j
n0(---------- line_0 ----------)n1(---------- line_1 ----------)n2
                                |                               |
                                n1->> load0_1_        _load1 <<-n2->> load1_1_
                                |      P10=30.0         P10=20.7       P10=4.3
                                |      Q10=5            Q10=5.7        Q10=2
                                |
                                |              y_mn=1e3-1e3j       
                                |              y_mm_half=1e-6+1e-6j
                                n1(---------- line_2 ----------)n3
                                                                |
                                                      _load2 <<-n3->> load2_1_
                                                        P10=20.7       P10=20
                                                        Q10=5.7        Q10=5.7  
```