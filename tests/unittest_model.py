# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:07:21 2022

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

@author: pyprg
"""
import unittest
import scipy.sparse
from numpy import inf
from pandas import DataFrame as DF
import context
from egrid import make_model
from egrid.builder import (
    Slacknode, Branch, Injection,
    make_data_frames, create_objects)
from egrid.model import Model, model_from_frames
from egrid.builder import Factor


_factor_test_string = """
                                  P10=7 Q10=4
                         n1 -->> inj1
                         ||
slack0 <------br0------> n1 <------br1------> n2 -->> inj2
               y_tr=1µ+1µj          y_tr=1µ+1µj        P10=1 Q10=.3
               y_lo=1k-1kj          y_lo=1k-1kj

# >> FACTORS for INJECTIONS <<

#.  Deff(id=Pinj1 m=2 type=const step(0 1 2))
#.  Link(objid=inj1 id=Pinj1 part=p step(0 1 2))
#.  Deff(id=Qinj1 type=var step(0 1 2))
#.  Link(objid=inj1 id=Qinj1 part=q step(0 1 2))

# >> FACTORS for BRANCHTERMINALS <<

#.  Deff(
#.     id=br0_slack0 value=0 m=-.00625 n=1 min=-16 max=16
#.     is_discrete=True type=const step(0 1 2))
#.  Link(objid=br0 nodeid=slack0 id=br0_slack0 cls=Terminallink step(0 1 2))
#.  Deff(
#.     id=br1_n1 value=0 m=-.00625 n=1 min=-16 max=16
#.     is_discrete=True type=var step(0 1 2))
#.  Link(objid=br1 nodeid=n1 id=br1_n1 cls=Terminallink step(0 1 2))"""

class Make_data_Frames(unittest.TestCase):

    def test_make_data_frames(self):
        frames = make_data_frames(create_objects(_factor_test_string))
        rowcounts = {
            'Branch': 2, 'Slacknode': 1, 'Injection': 2, 'Factor': 12,
            'Injectionlink': 6, 'Terminallink': 6}
        self.assertTrue(
            all(rowcounts.get(table_name, 0)==len(table)
                for table_name, table in frames.items()),
            'row count matches expectations')

class Make_model(unittest.TestCase):

    def test_make_model_empty(self):
        model = make_model()
        self.assertIsNotNone(
            model,
            msg='make_model shall return an object')
        self.assertIsInstance(
            model,
            Model,
            msg='make_model shall return an instance of egrid.model.Model')
        frames = [
            model.nodes,
            model.slacks,
            model.injections,
            model.branchterminals,
            model.branchoutputs,
            model.injectionoutputs,
            model.pvalues,
            model.qvalues,
            model.ivalues,
            model.vvalues,
            model.branchtaps,
            model.factors,
            model.injection_factor_associations,
            model.terms,
            model.messages]
        empty_dataframe = ((isinstance(df, DF) and df.empty) for df in frames)
        self.assertTrue(all(empty_dataframe), 'empty data frames')
        self.assertEqual(model.shape_of_Y, (0,0), 'matrix shape is 0,0')
        self.assertEqual(model.count_of_slacks, 0, 'no slack')
        self.assertIsInstance(
            model.mnodeinj,
            scipy.sparse.csc_matrix,
            msg='mnodeinj is scipy.sparse.csc_matrix')
        self.assertEqual(
            model.mnodeinj.shape,
            (0,0),
            'shape of mnodeinj is 0,0')

    def test_make_model_slack(self):
        model = make_model(Slacknode('n0'), Branch('b0', 'n0', 'n1'))
        self.assertIsNotNone(model, 'make_model shall return an object')
        # self.assertEqual(len(model.messages), 1, 'one error')
        self.assertEqual(len(model.slacks), 1, 'one slack node')
        self.assertEqual(model.slacks.id_of_node[0], 'n0', 'slack at node n0')

    def test_make_model_slack2(self):
        model = make_model([[[Slacknode('n0'), Branch('b0', 'n0', 'n1')]]])
        self.assertIsNotNone(model, 'make_model shall return an object')
        # self.assertEqual(len(model.messages), 1, 'one error')
        self.assertEqual(len(model.slacks), 1, 'one slack node')
        self.assertEqual(model.slacks.id_of_node[0], 'n0', 'slack at node n0')

    def test_make_model_slack3(self):
        model = make_model('n0\nslack=True')
        self.assertIsNotNone(model, 'make_model shall return an object')
        # self.assertEqual(len(model.messages), 4, 'four errors')

    def test_make_minimal_model(self):
        model = make_model(
            Slacknode('n0'),
            Branch('line_0', 'n0', 'n1'),
            Injection('load_0', 'n1'))
        self.assertIsNotNone(model, 'make_model shall return an object')
        # self.assertEqual(len(model.messages), 0, 'no errors')
        self.assertEqual(len(model.branchterminals), 2, 'two branch terminals')
        inf_cx = complex(inf, inf)
        self.assertTrue(all(inf_cx==y for y in model.branchterminals.y_lo))
        self.assertEqual(
            model.shape_of_Y, (1,1), 'one pfc node (branch without impedance)')

    def test_make_minimal_model2(self):
        model = make_model(
            ' slack=True                  P=10\n'
            'n0(-----line_0------)n1-->> load_0_')
        self.assertIsNotNone(model, 'make_model shall return an object')
        # self.assertEqual(len(model.messages), 0, 'no errors')
        self.assertEqual(len(model.branchterminals), 2, 'two branch terminals')
        inf_cx = complex(inf, inf)
        self.assertTrue(all(inf_cx==y for y in model.branchterminals.y_lo))
        self.assertEqual(
            model.shape_of_Y, (1,1), 'one pfc node (branch without impedance)')

    def test_make_with_factors(self):
        model = make_model(_factor_test_string)
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(
            len(model.branchterminals), 4, 'four branch terminals')
        self.assertEqual(
            model.shape_of_Y, (3,3), 'three pfc nodes')

class Model_messages(unittest.TestCase):

    elements = [
        Slacknode('n0'),
        Branch('line_0', 'n0', 'n1'),
        Injection('load_0', 'n1')]

    def test_wrong_object(self):
        model = make_model(self.elements, Factor(id='k'))
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.messages), 1, 'one error')

    def test_ignored(self):
        model = make_model(self.elements, 27)
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.messages), 1, 'one message')

class Model_from_frames(unittest.TestCase):

    def test_model_from_frames(self):
        # line_2 is a bridge as admittance is to high
        string = """
                y_tr=1e-6+1e-6j                 y_tr=1e-6+1e-6j
slack=True     y_lo=1e3-1e3j                   y_lo=1e3-1e3j
n0(---------- line_0 ----------)n1(---------- line_1 ----------)n2
                                |                               |
                                n1->> load0_1_        _load1 <<-n2->> load1_1_
                                |      P10=30.0         P10=20.7       P10=4.3
                                |      Q10=5            Q10=5.7        Q10=2
                                |
                                |              y_lo=1e8-1e8j
                                |              y_tr=1e-6+1e-6j
                                n1(---------- line_2 ----------)n3
                                                                |
                                                      _load2 <<-n3->> load2_1_
                                                        P10=20.7       P10=20
                                                        Q10=5.7        Q10=5.7
            """
        frames = make_data_frames(create_objects(string))
        model = model_from_frames(frames)
        self.assertIsInstance(
            model,
            Model,
            msg='make_data_frames shall return an instance of '
                'egrid.model.Model')
        self.assertEqual(
            model.shape_of_Y,
            (3, 3),
            'model.shape_of_Y shall be (3, 3)')

if __name__ == '__main__':
    unittest.main()