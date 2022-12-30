# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:07:21 2022

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

@author: pyprg
"""
import unittest
from src.egrid import make_model
from src.egrid.builder import (
    Slacknode, Branch, make_data_frames, create_objects)
from src.egrid.model import Model, model_from_frames

class Make_model(unittest.TestCase):

    def test_make_model_empty(self):
        self.assertIsNotNone(
            make_model(),
            msg='make_model shall return an object')

    def test_make_model_empty2(self):
        self.assertIsInstance(
            make_model(),
            Model,
            msg='make_model shall return an instance of egrid.model.Model')

    def test_make_model_slack(self):
        model = make_model(Slacknode('n0'), Branch('b0', 'n0', 'n1'))
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.errormessages), 1, 'one error')
        self.assertEqual(len(model.slacks), 1, 'one slack node')
        self.assertEqual(model.slacks.id_of_node[0], 'n0', 'slack at node n0')

    def test_make_model_slack2(self):
        model = make_model([[[Slacknode('n0'), Branch('b0', 'n0', 'n1')]]])
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.errormessages), 1, 'one error')
        self.assertEqual(len(model.slacks), 1, 'one slack node')
        self.assertEqual(model.slacks.id_of_node[0], 'n0', 'slack at node n0')

    def test_make_model_slack3(self):
        model = make_model('n0\nslack=True')
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.errormessages), 4, 'four errors')

class Model_from_frames(unittest.TestCase):

    def test_model_from_frames(self):
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