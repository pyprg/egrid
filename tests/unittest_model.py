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
from src.egrid.builder import Slacknode
from src.egrid.model import Model

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
        model = make_model(Slacknode('n0'))
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.errormessages), 0, 'no errors')
        self.assertEqual(len(model.slacks), 1, 'one slack node')
        self.assertEqual(model.slacks.id_of_node[0], 'n0', 'slack at node n0')
        
    def test_make_model_slack2(self):
        model = make_model([[[Slacknode('n0')]]])
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.errormessages), 0, 'no errors')
        self.assertEqual(len(model.slacks), 1, 'one slack node')
        self.assertEqual(model.slacks.id_of_node[0], 'n0', 'slack at node n0')
        
    def test_make_model_slack3(self):
        model = make_model('n0\nslack=True')
        self.assertIsNotNone(model, 'make_model shall return an object')
        self.assertEqual(len(model.errormessages), 1, 'one error')

if __name__ == '__main__':
    unittest.main()