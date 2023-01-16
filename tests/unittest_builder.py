# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:10:58 2022

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
import context
from egrid.builder import (make_objects, create_objects,
    Slacknode, PValue, QValue, Output, IValue, Branch, Injection, Message)

_EMPTY_DICT = {}

class Make_objects(unittest.TestCase):
    
    def test_make_objects_one_node(self):
        res = [*make_objects(('node', 'slack', (), {}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        self.assertIsInstance(
            res[0], Message, "make_objects shall return one message")
    
    def test_make_objects_slacknode(self):
        res = [*make_objects(('node', 'slack', ('adj'), {}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        slacknode = res[0]
        self.assertIsInstance(
            slacknode, 
            Slacknode, 
            "make_objects shall return one instance of Slacknode")
        self.assertEqual(
            slacknode.V, 
            Slacknode._field_defaults['V'], 
            'voltage attribute of slacknode shall have default value')
    
    def test_make_objects_slacknode2(self):
        res = [*make_objects(('node', 'n0', ('adj'), {'slack': 'True'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        slacknode = res[0]
        self.assertIsInstance(
            slacknode, 
            Slacknode, 
            "make_objects shall return one instance of Slacknode")
        self.assertEqual(
            slacknode.V, 
            Slacknode._field_defaults['V'], 
            'voltage attribute of slacknode shall have default value')
    
    def test_make_objects_slacknode3(self):
        res = [*make_objects(
            ('node', 'n0', ('adj'), {'slack': 'True', 'V': '0.97-0.2j'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        slacknode = res[0]
        self.assertIsInstance(
            slacknode, 
            Slacknode, 
            "make_objects shall return one instance of Slacknode")
        self.assertEqual(
            slacknode.V, 
            0.97-0.2j, 
            'voltage attribute of slacknode shall equal 0.97-0.2j')
    
    def test_make_objects_slacknode4(self):
        res = [*make_objects(
            ('node', 'n0', ('adj'), {'slack': 'True', 'V': '0.97-j0.2'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        self.assertIsInstance(
            res[0], Message, "make_objects shall return one message")
    
    def test_make_edge_objects_PQ(self):
        res = [*make_objects(
            ('edge', ('n0', 'line0'), {'P': '4.6', 'Q': '2.4'}))]
        self.assertEqual(
            len(res), 3, "make_objects shall return two object")
        self.assertIsInstance(
            res[0], 
            (PValue, QValue, Output), 
            "make_objects shall return instances of P/QValue and Output")
        self.assertIsInstance(
            res[1], 
            (PValue, QValue, Output), 
            "make_objects shall return instances of P/QValue and Output")
        self.assertIsInstance(
            res[2], 
            (PValue, QValue, Output), 
            "make_objects shall return instances of P/QValue and Output")
    
    def test_make_edge_objects_P(self):
        res = [*make_objects(
            ('edge', ('n0', 'line0'), {'P': '4.6'}))]
        self.assertEqual(
            len(res), 2, "make_objects shall return two object")
        self.assertIsInstance(
            res[0], 
            (PValue, Output), 
            "make_objects shall return instances of PValue and Output")
        self.assertIsInstance(
            res[1], 
            (PValue, Output), 
            "make_objects shall return instances of PValue and Output")
    
    def test_make_edge_objects_Q(self):
        res = [*make_objects(
            ('edge', ('n0', 'line0'), {'Q': '4.6'}))]
        self.assertEqual(
            len(res), 2, "make_objects shall return two object")
        self.assertIsInstance(
            res[0], 
            (QValue, Output), 
            "make_objects shall return instances of QValue and Output")
        self.assertIsInstance(
            res[1], 
            (QValue, Output), 
            "make_objects shall return instances of QValue and Output")
    
    def test_make_edge_objects_I(self):
        res = [*make_objects(
            ('edge', ('n0', 'line0'), {'I': '4.0'}))]
        self.assertEqual(
            len(res), 2, "make_objects shall return two object")
        self.assertIsInstance(
            res[0], 
            (IValue, Output), 
            "make_objects shall return instances of IValue and Output")
        self.assertIsInstance(
            res[1], 
            (IValue, Output), 
            "make_objects shall return instances of IValue and Output")
        
    def test_make_edge_objects_branch(self):
        res = [*make_objects(
            ('node', 'mybranch', ('n0', 'n1'), _EMPTY_DICT))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        branch = res[0]
        self.assertIsInstance(
            branch, 
            Branch, 
            "make_objects shall return an instance of Branch")
        self.assertEqual(
            branch.id, 
            'mybranch', 
            "id of branch shall be 'mybranch'")
        self.assertEqual(
            branch.y_lo, 
            Branch._field_defaults['y_lo'], 
            "y_lo of branch shall have default value")
        self.assertEqual(
            branch.y_tr, 
            Branch._field_defaults['y_tr'], 
            "y_tr of branch shall have default value")
        
    def test_make_edge_objects_branch2(self):
        res = [*make_objects(
            ('node', 
              'mybranch', 
              ('n0', 'n1'), 
              {'y_lo':'1+2j', 'y_tr': '3+7j'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        branch = res[0]
        self.assertEqual(
            branch.id, 
            'mybranch', 
            "id of branch shall be 'mybranch'")
        self.assertEqual(
            branch.y_lo, 
            1+2j, 
            "value of y_lo shall be 1+2j")
        self.assertEqual(
            branch.y_tr, 
            3+7j, 
            "value of y_tr shall be 3+7j")
        
    def test_make_edge_objects_injection(self):
        res = [*make_objects(
            ('node', 'myinjection', ('n0',), _EMPTY_DICT))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        injection = res[0]
        self.assertIsInstance(
            injection, 
            Injection, 
            "make_objects shall return an instance of Injection")
        self.assertEqual(
            injection.id, 
            'myinjection', 
            "id of branch shall be 'myinjection'")
        self.assertEqual(
            injection.P10, 
            Injection._field_defaults['P10'], 
            "P10 of injection shall have default value")
        self.assertEqual(
            injection.Q10, 
            Injection._field_defaults['Q10'], 
            "Q10 of injection shall have default value")
        self.assertEqual(
            injection.Exp_v_p, 
            Injection._field_defaults['Exp_v_p'], 
            "Exp_v_p of injection shall have default value")
        self.assertEqual(
            injection.Exp_v_q, 
            Injection._field_defaults['Exp_v_q'], 
            "Exp_v_q of injection shall have default value")
        
    def test_make_edge_objects_injection2(self):
        res = [*make_objects(
            ('node', 
              'myinjection', 
              ('n0',), 
              {'P10': '1', 'Q10': '2', 'Exp_v_p': '3', 'Exp_v_q': '7'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        injection = res[0]
        self.assertIsInstance(
            injection, 
            Injection, 
            "make_objects shall return an instance of Injection")
        self.assertEqual(
            injection.id, 'myinjection', "id of branch shall be 'myinjection'")
        self.assertEqual(
            injection.P10, 1.0, "P10 shall equal 1.0")
        self.assertEqual(
            injection.Q10, 2.0, "Q10 shall equal 2.0")
        self.assertEqual(
            injection.Exp_v_p, 3.0, "Exp_v_p shall equal 3.0")
        self.assertEqual(
            injection.Exp_v_q, 7.0, "Exp_v_q shall equal 7.0")
        
    def test_make_edge_objects_message(self):
        # error no neighbours
        res = [*make_objects(('node', 'myid', (), {}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        obj = res[0]
        self.assertIsInstance(
            obj, Message, "make_objects shall return an instance of Message")
        # error P10 no float
        res = [*make_objects(('node', 'myid', ('n1',), {'P10':'1.+.4j'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        obj = res[0]
        self.assertIsInstance(
            obj, Message, "make_objects shall return an instance of Message")
        # error P10 no float
        res = [*make_objects(('node', 'myid', ('n1',), {'P10':'hallo'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        obj = res[0]
        self.assertIsInstance(
            obj, Message, "make_objects shall return an instance of Message")
        # error P10 no float
        res = [*make_objects(('node', 'myid', ('n1',), {'P10':'True'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        obj = res[0]
        self.assertIsInstance(
            obj, Message, "make_objects shall return an instance of Message")
        # error Q10 no float
        res = [*make_objects(('node', 'myid', ('n1',), {'Q10':'1.+.4j'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        obj = res[0]
        self.assertIsInstance(
            obj, Message, "make_objects shall return an instance of Message")
        # error Exp_v_p no float
        res = [*make_objects(('node', 'myid', ('n1',), {'Exp_v_p':'1.+.4j'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        obj = res[0]
        self.assertIsInstance(
            obj, Message, "make_objects shall return an instance of Message")
        # error Exp_v_q no float
        res = [*make_objects(('node', 'myid', ('n1',), {'Exp_v_q':'1.+.4j'}))]
        self.assertEqual(
            len(res), 1, "make_objects shall return one object")
        obj = res[0]
        self.assertIsInstance(
            obj, Message, "make_objects shall return an instance of Message")

class Create_objects(unittest.TestCase):
    
    def test_empty_string(self):
        res = [*create_objects('')]
        self.assertEqual(res, [], 'is empty')
        
    def test_empty_iterable(self):
        self.assertEqual(
            [*create_objects([])], 
            [], 
            'is empty')
        self.assertEqual(
            [*create_objects([(),()])], 
            [], 
            'is empty')
    
    def test_single_node(self):
        msg, *_ = [*create_objects('n0')]
        self.assertIsInstance(msg, Message, 'create object returns a message')
        self.assertEqual(msg.level, 1, 'create_objects returns warning')
        self.assertTrue(
            msg.message.startswith('ignoring object'),
            'create object ignores object')
    
    def test_none_node_string(self):
        msg, *_ = [*create_objects('string')]
        self.assertIsInstance(msg, Message, 'create object returns a message')
        self.assertEqual(msg.level, 1, 'create_objects returns warning')
        self.assertTrue(
            msg.message.startswith('ignoring object'),
            'create object ignores object')
    
    def test_two_none_node_strings(self):
        _, msg, __ = [*create_objects('object_a object_b')]
        self.assertIsInstance(msg, Message, 'create object returns a message')
        self.assertEqual(
            msg.level, 2, 'create_objects returns an error message')
        self.assertTrue(
            msg.message.startswith('Error'),
            'create object issues an error message')
    
    def test_node_injection(self):
        res = [*create_objects('node injection')]
        self.assertAlmostEqual(
            len(res), 1, 'create_objects returns one object')
        self.assertIsInstance(
            res[0], 
            Injection, 
            'create_objects returns an instance of Injection')
    
    def test_node_branch_node(self):
        res = [*create_objects('node_A branch node_B')]
        self.assertAlmostEqual(
            len(res), 1, 'create_objects returns one object')
        self.assertIsInstance(
            res[0], 
            Branch, 
            'create_objects returns an instance of Branch')
        branch = res[0]
        self.assertEqual(
            branch.id, 'branch', 'the ID of the branch is "branch"')
        self.assertEqual(
            branch.id_of_node_A, 'node_A', 'id_of_node_A is node_A')
        self.assertEqual(
            branch.id_of_node_B, 'node_B', 'id_of_node_B is node_B')
    
    def test_not_processed_type(self):
        msg, *_ = [*create_objects(27)]
        self.assertIsInstance(msg, Message, 'create_objects returns a message')
        self.assertEqual(msg.level, 1, 'create_objects returns warning')
        self.assertTrue(
            msg.message.startswith('wrong type'),
            'create_objects ignores object')
    
    def test_pass_branch(self):
        branch = Branch('b', 'n0', 'n1')
        res, *_ = [*create_objects(branch)]
        self.assertEqual(
            res, branch, 'create_objects returns Branch instance')

if __name__ == '__main__':
    unittest.main()
