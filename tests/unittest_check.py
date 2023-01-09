# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:05:27 2023

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

import unittest
from pandas import DataFrame
from src.egrid.builder import make_data_frames
from src.egrid._types import (
    Slacknode, PValue, QValue, Output, IValue, Vvalue, 
    Branch, Injection, 
    Defk, Link)
from src.egrid.check import (
    check_numbers, check_factor_links, check_batch_links)

_elements = [
    Slacknode('n0'), 
    Branch('line_0', 'n0', 'n1'), 
    Injection('load_0', 'n1')]

class Check_numbers(unittest.TestCase):

    def test_without_objects(self):
        """3 messages for an empty model."""
        frames = make_data_frames([])
        self.assertIsInstance(
            frames, 
            dict, 
            'make_data_frames shall return an instance of dict')
        self.assertIsInstance(
            frames['Message'],
            DataFrame,
            'frames["Message"] is a pandas.DataFrame')
        msgs = frames['Message']
        self.assertEqual(
            len(msgs),
            0,
            'frames["Message"] is empty')
        messages = [*check_numbers(frames)]
        self.assertEqual(
            len(messages), 
            3,
            'check_numbers yields 3 messages')

class Check_factors_links(unittest.TestCase):
    
    def test_defk_with_link(self):
        """Define a load factor and link it to an injection. No messages."""
        frames = make_data_frames(
            _elements + [Defk(id='k'), Link('load_0', 'p', 'k')])
        self.assertIsInstance(
            frames, dict, 'make_data_frames shall return an instance of dict')
        self.assertEqual(len(frames['Message']), 0, 'no error')
        self.assertEqual(len(frames['Injection']), 1, 'one injection')
        self.assertEqual(
            len(frames['KInjlink']), 1, 'one instance of KInjlink')
        self.assertEqual(
            len(frames['Loadfactor']), 1, 'one instance of Loadfactor')
        messages = [*check_factor_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_factors_links yields no message')

    def test_defk_without_link(self):
        """Message for unlinked load factor."""
        frames = make_data_frames([Defk('k')])
        self.assertIsInstance(
            frames, 
            dict, 
            'make_data_frames shall return an instance of dict')
        self.assertIsInstance(
            frames['Loadfactor'],
            DataFrame,
            'frames["Loadfactor"] is a pandas.DataFrame')
        loadfactors = frames['Loadfactor']
        self.assertEqual(
            len(loadfactors),
            1,
            'frames["Loadfactor"] has one row')
        messages = [*check_factor_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_factors_links yields 1 message')
        
    def test_defk_without_link2(self):
        """Message for unlinked load factor."""
        frames = make_data_frames(
            _elements + [Defk(id='k')])
        self.assertIsInstance(
            frames, 
            dict, 
            'make_data_frames shall return an instance of dict')
        self.assertIsInstance(
            frames['Loadfactor'],
            DataFrame,
            'frames["Loadfactor"] is a pandas.DataFrame')
        loadfactors = frames['Loadfactor']
        self.assertEqual(
            len(loadfactors),
            1,
            'frames["Loadfactor"] has one row')
        messages = [*check_factor_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_factors_links yields 1 message')
        
    def test_defk_with_invalid_link(self):
        """Message for link with invalid reference to injection."""
        frames = make_data_frames(
            _elements + [Defk(id='k'), Link('invalid_id', 'p', id='k')])
        self.assertIsInstance(
            frames, 
            dict, 
            'make_data_frames shall return an instance of dict')
        self.assertIsInstance(
            frames['Loadfactor'],
            DataFrame,
            'frames["Loadfactor"] is a pandas.DataFrame')
        loadfactors = frames['Loadfactor']
        self.assertEqual(
            len(loadfactors),
            1,
            'frames["Loadfactor"] has one row')
        messages = [*check_factor_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_factors_links yields 1 message')
        
    def test_defk_with_invalid_link2(self):
        """Message for link with invalid reference to factor."""
        frames = make_data_frames(
            _elements 
            + [Defk(id='k'), Link('load_0', 'p', id='invalid_factor_id')])
        self.assertIsInstance(
            frames, 
            dict, 
            'make_data_frames shall return an instance of dict')
        self.assertIsInstance(
            frames['Loadfactor'],
            DataFrame,
            'frames["Loadfactor"] is a pandas.DataFrame')
        loadfactors = frames['Loadfactor']
        self.assertEqual(
            len(loadfactors),
            1,
            'frames["Loadfactor"] has one row')
        messages = [*check_factor_links(frames)]
        self.assertEqual(
            len(messages), 
            2,
            'check_factors_links yields 2 messages')
    
    def test_link_with_invalid_factor_reference(self):
        """link with invalid reference to load scaling factor"""
        frames = make_data_frames(
            _elements + [Link('load_0', 'p', 'k')])
        self.assertIsInstance(
            frames, 
            dict, 
            'make_data_frames shall return an instance of dict')
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Injection']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 injection')
        self.assertIsInstance(
            frames.get('KInjlink'),
            DataFrame,
            'frames["KInjlink"] exists')
        links = frames['KInjlink']
        self.assertEqual(
            len(links), 
            1,
            'make_data_frames returns 1 link')
        messages = [*check_factor_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_factors_links yields 1 message')

class Check_batch_links(unittest.TestCase):
    
    def test_single_injection_output(self):
        """invalid id_of_batch, output at injection"""
        frames = make_data_frames(
            _elements 
            + [Output(id_of_batch='at_load_0', id_of_device='load_0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Injection']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 injection')
        self.assertIsInstance(
            frames.get('IValue'),
            DataFrame,
            'frames["IValue"] exists')
        ivalues = frames['IValue']
        self.assertEqual(
            len(ivalues), 
            0,
            'make_data_frames returns no I-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_batch_links yields 1 message')
    
    def test_single_branch_output(self):
        """invalid id_of_batch, Output at branch"""
        frames = make_data_frames(
            _elements 
            + [Output(
                id_of_batch='at_line_0', 
                id_of_device='line_0',
                id_of_node='n0')])
        self.assertIsInstance(
            frames.get('Branch'),
            DataFrame,
            'frames["Branch"] exists')
        injections = frames['Branch']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 branch')
        self.assertIsInstance(
            frames.get('IValue'),
            DataFrame,
            'frames["IValue"] exists')
        ivalues = frames['IValue']
        self.assertEqual(
            len(ivalues), 
            0,
            'make_data_frames returns no I-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_batch_links yields 1 message')
    
    def test_valid_ivalue_at_injection(self):
        """valid configuration, no messages, IValue at injection"""
        frames = make_data_frames(
            _elements 
            + [IValue(id_of_batch='at_load_0', I=7), 
                Output(id_of_batch='at_load_0', id_of_device='load_0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Injection']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 injection')
        self.assertIsInstance(
            frames.get('IValue'),
            DataFrame,
            'frames["IValue"] exists')
        ivalues = frames['IValue']
        self.assertEqual(
            len(ivalues), 
            1,
            'make_data_frames returns 1 I-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_batch_links yields no message')
    
    def test_valid_pvalue_at_injection(self):
        """valid configuration, no messages, PValue at injection"""
        frames = make_data_frames(
            _elements 
            + [PValue(id_of_batch='at_load_0', P=7), 
                Output(id_of_batch='at_load_0', id_of_device='load_0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Injection']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 injection')
        self.assertIsInstance(
            frames.get('PValue'),
            DataFrame,
            'frames["PValue"] exists')
        pvalues = frames['PValue']
        self.assertEqual(
            len(pvalues), 
            1,
            'make_data_frames returns 1 P-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_batch_links yields no message')
    
    def test_valid_qvalue_at_injection(self):
        """valid configuration, no messages, QValue at injection"""
        frames = make_data_frames(
            _elements 
            + [QValue(id_of_batch='at_load_0', Q=7), 
                Output(id_of_batch='at_load_0', id_of_device='load_0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Injection']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 injection')
        self.assertIsInstance(
            frames.get('QValue'),
            DataFrame,
            'frames["QValue"] exists')
        qvalues = frames['QValue']
        self.assertEqual(
            len(qvalues), 
            1,
            'make_data_frames returns 1 Q-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_batch_links yields no message')
    
    def test_valid_ivalue_at_branch(self):
        """valid configuration, no messages, IValue at branch"""
        frames = make_data_frames(
            _elements 
            + [IValue(id_of_batch='at_line_0', I=7), 
                Output(
                    id_of_batch='at_line_0', 
                    id_of_device='line_0',
                    id_of_node='n0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Branch']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 branch')
        self.assertIsInstance(
            frames.get('IValue'),
            DataFrame,
            'frames["IValue"] exists')
        ivalues = frames['IValue']
        self.assertEqual(
            len(ivalues), 
            1,
            'make_data_frames returns 1 I-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_batch_links yields no message')
    
    def test_valid_pvalue_at_branch(self):
        """valid configuration, no messages, PValue at branch"""
        frames = make_data_frames(
            _elements 
            + [PValue(id_of_batch='at_line_0', P=7), 
                Output(
                    id_of_batch='at_line_0', 
                    id_of_device='line_0',
                    id_of_node='n0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Branch']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 branch')
        self.assertIsInstance(
            frames.get('PValue'),
            DataFrame,
            'frames["PValue"] exists')
        pvalues = frames['PValue']
        self.assertEqual(
            len(pvalues), 
            1,
            'make_data_frames returns 1 P-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_batch_links yields no message')
    
    def test_valid_qvalue_at_branch(self):
        """valid configuration, no messages, QValue at branch"""
        frames = make_data_frames(
            _elements 
            + [QValue(id_of_batch='at_line_0', Q=7), 
                Output(
                    id_of_batch='at_line_0', 
                    id_of_device='line_0',
                    id_of_node='n0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Branch']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 branch')
        self.assertIsInstance(
            frames.get('QValue'),
            DataFrame,
            'frames["QValue"] exists')
        qvalues = frames['QValue']
        self.assertEqual(
            len(qvalues), 
            1,
            'make_data_frames returns 1 Q-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_batch_links yields no message')
    
    def test_valid_vvalue_at_node(self):
        """valid configuration, no messages, Vvalue at node"""
        frames = make_data_frames(
            _elements 
            + [Vvalue(id_of_node='n0', V=.99)])
        self.assertIsInstance(
            frames.get('Vvalue'),
            DataFrame,
            'frames["Vvalue"] exists')
        vvalues = frames['Vvalue']
        self.assertEqual(
            len(vvalues), 
            1,
            'make_data_frames returns 1 V-value')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            0,
            'check_batch_links yields no message')
    
    def test_injection_output_with_invalid_injection_reference(self):
        """invalid configuration, referenced injection does not exist, 
        1 messages"""
        frames = make_data_frames(
            _elements 
            + [IValue(id_of_batch='at_load_0', I=7), 
                Output(id_of_batch='at_load_0', id_of_device='invalid')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Injection']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 injection')
        self.assertIsInstance(
            frames.get('IValue'),
            DataFrame,
            'frames["IValue"] exists')
        ivalues = frames['IValue']
        self.assertEqual(
            len(ivalues), 
            1,
            'make_data_frames returns 1 I-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_batch_links yields 1 message')

    def test_branch_output_with_invalid_branch_reference(self):
        """invalid configuration, referenced branch does not exist, 
        1 messages"""
        frames = make_data_frames(
            _elements 
            + [IValue(id_of_batch='at_line_0', I=7), 
                Output(
                    id_of_batch='at_line_0', 
                    id_of_device='invalid',
                    id_of_node='n0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Branch']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 branch')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_batch_links yields no message')
        
    def test_branch_output_with_invalid_node_reference(self):
        """invalid configuration, referenced node does not exist, 
        1 messages"""
        frames = make_data_frames(
            _elements 
            + [IValue(id_of_batch='at_line_0', I=7), 
                Output(
                    id_of_batch='at_line_0', 
                    id_of_device='line_0',
                    id_of_node='invalid')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Branch']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 branch')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_batch_links yields no message')
    
    def test_ivalue_with_invalid_batch_reference(self):
        """valid configuration, 1 messages"""
        frames = make_data_frames(
            _elements 
            + [IValue(id_of_batch='at_load_1', I=7), 
                Output(id_of_batch='at_load_0', id_of_device='load_0')])
        self.assertIsInstance(
            frames.get('Injection'),
            DataFrame,
            'frames["Injection"] exists')
        injections = frames['Injection']
        self.assertEqual(
            len(injections), 
            1,
            'make_data_frames returns 1 injection')
        self.assertIsInstance(
            frames.get('IValue'),
            DataFrame,
            'frames["IValue"] exists')
        ivalues = frames['IValue']
        self.assertEqual(
            len(ivalues), 
            1,
            'make_data_frames returns 1 I-value')
        self.assertIsInstance(
            frames.get('Output'),
            DataFrame,
            'frames["Output"] exists')
        outputs = frames['Output']
        self.assertEqual(
            len(outputs), 
            1,
            'make_data_frames returns 1 output')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            2,
            'check_batch_links yields 1 message')
    
    def test_vvalue_with_invalid_node_reference(self):
        """valid configuration, no messages, Vvalue at node"""
        frames = make_data_frames(
            _elements 
            + [Vvalue(id_of_node='n', V=.99)])
        self.assertIsInstance(
            frames.get('Vvalue'),
            DataFrame,
            'frames["Vvalue"] exists')
        vvalues = frames['Vvalue']
        self.assertEqual(
            len(vvalues), 
            1,
            'make_data_frames returns 1 V-value')
        messages = [*check_batch_links(frames)]
        self.assertEqual(
            len(messages), 
            1,
            'check_batch_links yields no message')
    
if __name__ == '__main__':
    unittest.main()
