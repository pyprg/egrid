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
from numpy.testing import assert_array_equal
import pandas as pd
from egrid.builder import (make_objects, create_objects, make_data_frames,
    Slacknode, PValue, QValue, Output, IValue, Branch, Injection,
    Message, Defk, Klink, Tlink, Vlimit, Defoterm)

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
        self.assertEqual(
            len(res), 1, 'create_objects returns one object')
        self.assertIsInstance(
            res[0],
            Injection,
            'create_objects returns an instance of Injection')

    def test_node_branch_node(self):
        res = [*create_objects('node_A branch node_B')]
        self.assertEqual(
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
        res, *_ = create_objects(branch)
        self.assertEqual(
            res, branch, 'create_objects returns Branch instance')

    def test_create_message_empty_instruction_line(self):
        res = [*create_objects('#.')]
        self.assertEqual(
            len(res), 0, 'create_objects returns no object')

    def test_create_instruction(self):
        res = [*create_objects(
            ['#.   Message(message=hallo, level=0)',
             '#.   Defk(id=kp)'])]
        self.assertEqual(
            len(res), 2, 'create_objects returns two objects')
        self.assertEqual(
            res,
            [Message(message='hallo', level=0),
              Defk(id=('kp',))],
            'create_objects creates instances of Message, Defk')

    def test_create_instruction2(self):
        """remove quotes from strings"""
        res = [*create_objects(
            ['#.   Message(message="hallo", level=0)',
              '#.   Defk(id="kp")'])]
        self.assertEqual(
            len(res), 2, 'create_objects returns two objects')
        self.assertEqual(
            res,
            [Message(message='hallo', level=0),
              Defk(id=('kp',))],
            'create_objects creates instances of Message, Defk')

    def test_create_injection(self):
        res = [*create_objects(
            ['#. Klink(id_of_injection=hallo part=p id_of_factor=myid)'])]
        self.assertEqual(
            len(res), 1, 'create_objects returns one object')
        self.assertEqual(
            res,
            [Klink(
                part=('p',),
                id_of_injection=('hallo',),
                id_of_factor=('myid',))],
            'create_objects creates instances of Link')

    def test_create_terminallink(self):
        res = [*create_objects(
            ['#. Tlink(id_of_branch=hallo id_of_node=node0 id_of_factor=myid '
              'step=(1 2))'])]
        self.assertEqual(
            len(res), 1, 'create_objects returns one object')
        self.assertEqual(
            res,
            [Tlink(
                id_of_branch=('hallo',),
                id_of_node=('node0',),
                id_of_factor=('myid',),
                step=(1,2))],
            'create_objects creates instances of Tlink')

    def test_create_defoterm(self):
        res = [*create_objects(['#. Defoterm()'])]
        self.assertEqual(len(res), 1, 'create_objects returns one object')
        self.assertEqual(
            res, [Defoterm()], 'create_objects creates instances of Defoterm')

class Make_data_frames(unittest.TestCase):
    """objects input"""

    def test_empty(self):
        frames = make_data_frames()
        self.assertEqual(
            len(frames), 14, 'make_data_frames creates 13 items')
        self.assertTrue(
            all(isinstance(df, pd.DataFrame) for key, df in frames.items()),
            'all values are pandas.DataFrames')
        self.assertTrue(
            all(df.empty for key, df in frames.items()),
            'all dataframes are empty')

    def test_term(self):
        frames = make_data_frames([Defoterm()])
        term = frames.get('Term')
        self.assertIsInstance(term, pd.DataFrame)
        row = term.iloc[0]
        self.assertEqual(row.id, '0')
        assert_array_equal(row.args, Defoterm._field_defaults['args'])
        self.assertEqual(row.fn, Defoterm._field_defaults['fn'])
        self.assertEqual(row.step, Defoterm._field_defaults['step'])

    def test_vlimit(self):
        frames = make_data_frames([Vlimit(id_of_node='n_0')])
        vlimit = frames['Vlimit'].loc[0]
        self.assertEqual(vlimit.id_of_node, 'n_0')
        self.assertAlmostEqual(vlimit['min'], 0.9)
        self.assertAlmostEqual(vlimit['max'], 1.1)
        self.assertEqual(vlimit.step, -1)

class Make_data_frames2(unittest.TestCase):
    """string input"""

    def test_empty(self):
        objs = create_objects(
            ['#.'])
        frames = make_data_frames(objs)
        self.assertEqual(len(frames.items()), 14)
        for k in [
                'Branch', 'Slacknode', 'Injection', 'Output', 'PValue',
                'QValue', 'IValue', 'Vvalue', 'Vlimit', 'Term', 'Message',
                'Factor', 'Injectionlink', 'Terminallink']:
            df = frames[k]
            self.assertTrue(df.empty)

    def test_injlink_in_footer(self):
        objs = create_objects(
            ['#. Klink(id_of_injection=hallo part=p id_of_factor=f0',
             '#.      step=(1 2))'])
        frames = make_data_frames(objs)
        self.assertEqual(
            len(frames['Injectionlink']),
            2,
            'two rows in table Injectionlink')

    def test_injlink2_in_footer(self):
        objs = create_objects(
            ['#. Klink(id_of_injection=hallo part=(p q) id_of_factor=(f0 f1)',
             '#.      step=(1 2))'])
        frames = make_data_frames(objs)
        self.assertEqual(
            len(frames['Injectionlink']),
            4,
            'four rows in table Injectionlink')

    def test_terminallink_in_footer(self):
        objs = create_objects(
            ['#. Tlink(id_of_branch=hallo id_of_node=node0 id_of_factor=f0',
             '#.      step=(1 2))'])
        frames = make_data_frames(objs)
        self.assertEqual(len(frames['Terminallink']),
            2,
            'two rows in table Terminallink')

    def test_terminallink2_in_footer(self):
        objs = create_objects(
            ['#. Tlink(id_of_branch=(br0 br1) id_of_node=(n0 n1)',
             '#.       id_of_factor=f0 step=(1 2))'])
        frames = make_data_frames(objs)
        self.assertEqual(
            len(frames['Terminallink']), 4, 'four rows in table Terminallink')

    def test_wrong_class(self):
        objs = create_objects(
            ['#. Link(objid=(br0 br1) nodeid=(n0 n1) id=f0',
             '#.      step=(1 2))'])
        frames = make_data_frames(objs)
        self.assertEqual(len(frames['Message']), 1, 'one error message')
        self.assertTrue(frames['Terminallink'].empty)
        self.assertTrue(frames['Injectionlink'].empty)

    def test_defvl(self):
        objs = create_objects('#.Defvl(id_of_node=n_0)')
        frames = make_data_frames(objs)
        vlimit = frames['Vlimit'].loc[0]
        self.assertEqual(vlimit.id_of_node, 'n_0')
        self.assertAlmostEqual(vlimit['min'], 0.9)
        self.assertAlmostEqual(vlimit['max'], 1.1)
        self.assertEqual(vlimit.step, -1)

    def test_defvl_in_footer(self):
        objs = create_objects('#.Defvl(id_of_node(n0 n1) step(0 1 2))')
        frames = make_data_frames(objs)
        self.assertEqual(len(frames['Message']), 0, 'no error message')
        self.assertEqual(len(frames['Vlimit']), 6, 'Vlimit has six row')

    def test_vlimit_at_node(self):
        """Vlimit at node"""
        objs = create_objects(
            'n      inj\n'
            'Vlimit.min=.98\n'
            'Vlimit.step=1')
        frames = make_data_frames(objs)
        self.assertEqual(len(frames['Message']), 0, 'no error message')
        self.assertEqual(len(frames['Vlimit']), 1, 'Vlimit has one row')
        self.assertEqual(
            frames['Vlimit'].iloc[0][['id_of_node','min','step']].to_list(),
            ['n', 0.98, 1])

    def test_vlimit_in_footer(self):
        """Vlimit in footer"""
        objs = create_objects('#.Vlimit')
        frames = make_data_frames(objs)
        self.assertEqual(len(frames['Message']), 0, 'no error message')
        self.assertEqual(len(frames['Vlimit']), 1, 'Vlimit has one row')

    def test_P_at_node(self):
        """PValue at node"""
        objs = create_objects(
            'n       inj\n'
            '  P=10 P.direction=-1')
        frames = make_data_frames(objs)
        self.assertEqual(
            len(frames['Message']), 0, 'no error message')
        self.assertEqual(len(frames['PValue']), 1, 'Pvalue has one row')
        self.assertEqual(
            frames['PValue'].iloc[0]
            [['id_of_batch', 'P', 'direction', 'cost']].to_list(),
            ['n_inj', 10.0, -1.0, 0.0])

    def test_PValue_in_footer(self):
        """PValue in footer"""
        objs = create_objects(
            '#.PValue(id_of_batch=a P=12 direction=-1 cost=27)')
        frames = make_data_frames(objs)
        self.assertEqual(
            len(frames['Message']),
            0,
            'no error message')
        self.assertEqual(
            len(frames['PValue']),
            1,
            'Pvalue has one row')
        self.assertEqual(
            frames['PValue'].iloc[0]
            [['id_of_batch', 'P', 'direction', 'cost']].to_list(),
            ['a', 12.0, -1.0, 27.0])

if __name__ == '__main__':
    unittest.main()
