# egrid

Model of an electric distribution network for power flow calculation and 
state estimation. The model is an instance of namedtuple. Most of the fields
are pandas.DataFrames. Electric values are stored per unit.
The main function is get_model(*args). The arguments are of type 
    - Branch (line, series capacitor, transformer winding, transformer)
    - Slacknode
    - Injection (consumer, shunt capacitor, PQ/PV-generator, battery)
    - Output (indicates that measured flow (I or PQ) or a part thereof flows through the referenced terminal (device or node+device))
    - PQValue (measured active and reactive power)
    - IValue (measured electric current)
    - Vvalue (measured voltage or setpoint)
    - Branchtaps
    - Defk (definition of a scaling factor, for estimation)
    - Link (associates a scaling factor to a load)

Additionally, function get_model can consume network descriptions as multiline 
strings if package 'graphparser' is installed.
    
Branch models are PI-equivalent circuits. Active and reactive power of
injections have a dedicated voltage exponent.

Fields
------
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
    * .g_tot, float, conductance, g_mn + g_mm_half
    * .b_tot, float, susceptande, b_mn + b_mm_half
    * .g_mn, float, longitudinal conductance
    * .b_mn, float, longitudinal susceptance
    * .g_mm_half, float, transversal conductance devided by 2
    * .b_mm_half, float, transversal susceptance devided by 2
    * .side, str, 'A' | 'B', side of branch, first or second
branchoutputs: pandas.DataFrame
    * .id_of_batch, str, unique identifier of measurement point
    * .id_of_node, str, id of node connected to branch terminal
    * .id_of_branch, str, unique identifier of branch
    * .index_of_node, int, index of node connected to branch terminal
    * .index_of_branch, int, index of branch
injectionoutputs: pandas.DataFrame
    * .id_of_batch, str, unique identifier of measurement point 
    * .id_of_injection, str, unique identifier of injection
    * .index_of_injection, str, index of injection
pqvalues: pandas.DataFrame
    * .id_of_batch, unique identifier of measurement point
    * .P, float, active power
    * .Q, float, reactive power
    * .direction, float, -1: from device into node, 1: from node into device
ivalues: pandas.DataFrame
    * .id_of_batch, unique identifier of measurement point
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