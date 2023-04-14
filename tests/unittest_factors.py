# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:10:49 2023

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
import pandas as pd
from numpy.testing import assert_array_equal
from egrid.model import (
    _get_pfc_nodes, _prepare_branches, _get_branch_terminals, _add_bg)
from egrid.builder import (
    Factor, Injectionlink, Terminallink, Branch)
from egrid._types import (
    FACTORS, TERMINALLINKS, INJLINKS, BRANCHES)
from egrid.factors import make_factordefs

def _terminallink_frame(termlinks):
    terminallinks = (
        termlinks.copy()
        .set_index(['step', 'branchid', 'nodeid']))
    terminallinks.index.names = ['step', 'id_of_branch', 'id_of_node']
    return terminallinks

def _injectionlink_frame(injlinks):
    injectionlinks = (
        injlinks.set_index(['step', 'injid', 'part']))
    injectionlinks.index.names = ['step', 'id_of_injection', 'part']
    return injectionlinks

def _branchterminal_frame(branches, id_of_slacknodes):
    branches_ = branches
    branches_['is_bridge'] = False
    pfc_slack_count, node_count, pfc_nodes = _get_pfc_nodes(
        id_of_slacknodes, branches_)
    branches2 = _prepare_branches(branches_, pfc_nodes)
    return _get_branch_terminals(_add_bg(branches2))

class Make_factordefs(unittest.TestCase):

    # prepare empty pandas.DataFrame instances
    no_terminallinks = _terminallink_frame(TERMINALLINKS.copy())
    no_injectionlinks = _injectionlink_frame(INJLINKS.copy())
    no_branchterminals = _branchterminal_frame(
        BRANCHES.copy(), pd.Series([], name='id_of_node', dtype=str))
    no_factors = FACTORS.copy()

    def test_empty(self):
        """without data"""
        factordefs = make_factordefs(
            self.no_factors,
            self.no_terminallinks,
            self.no_injectionlinks,
            self.no_branchterminals)
        self.assertTrue(
            factordefs.gen_factor_data.empty,
            "no generic factors")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "no links from injections to generic factors")
        self.assertTrue(
            factordefs.gen_termfactor.empty,
            "no links from terminal to generic factors")
        self.assertTrue(
            all(factordefs.get_groups([idx]).empty
                for idx in range(-1, 5)),
            "make_factordefs shall not return any factor for any step")
        self.assertTrue(
            all(factordefs.get_injfactorgroups([idx]).empty
                for idx in range(-1, 5)),
            "make_factordefs shall not return any link "
            "from injection to factor for any step")

    def test_generic_scaling_factor(self):
        """one generic scaling factor"""
        factors = pd.DataFrame([Factor(id='kp')])
        injectionlinks = _injectionlink_frame(
            pd.DataFrame([Injectionlink(injid='i_0', part='p', id='kp')]))
        factordefs = make_factordefs(
            factors,
            self.no_terminallinks,
            injectionlinks,
            self.no_branchterminals)
        self.assertEqual(
            len(factordefs.gen_factor_data),
            1,
            "make_factordefs shall return one generic factor")
        self.assertEqual(
            len(factordefs.gen_injfactor),
            1,
            "make_factordefs shall return "
            "one link: injection -> generic factor")
        self.assertTrue(
            factordefs.gen_termfactor.empty,
            "make_factordefs shall not return "
            "any link: terminal -> generic factor")
        self.assertTrue(
            all(factordefs.get_groups([idx]).empty
                for idx in range(5)),
            "make_factordefs shall not return any factor for any step")
        self.assertTrue(
            all(factordefs.get_injfactorgroups([idx]).empty
                for idx in range(5)),
            "make_factordefs shall not return any link "
            "from injection to factor for any step")

    def test_scaling_factor_step0(self):
        """one scaling factor for step 0"""
        factors = pd.DataFrame([Factor(id='kp', step=0)])
        injectionlinks = _injectionlink_frame(
            pd.DataFrame(
                [Injectionlink(injid='i_0', part='p', id='kp', step=0)]))
        factordefs = make_factordefs(
            factors,
            self.no_terminallinks,
            injectionlinks,
            self.no_branchterminals)
        self.assertTrue(
            factordefs.gen_factor_data.empty,
            "make_factordefs shall not return any generic factor")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "no links from injections to generic factors")
        self.assertTrue(
            factordefs.gen_termfactor.empty,
            "make_factordefs shall not return "
            "any link from terminal to generic factor")
        assert_array_equal(
            [len(factordefs.get_groups([idx])) for idx in range(-1, 5)],
            [0, 1, 0, 0, 0, 0],
            err_msg="make_factordefs shall return one factor for step 0")
        assert_array_equal(
            [len(factordefs.get_injfactorgroups([idx]))
             for idx in range(-1, 5)],
            [0, 1, 0, 0, 0, 0],
            err_msg="make_factordefs shall return one link injection->factor "
            "for step 0")

    def test_generic_taps_factor(self):
        """one generic taps (terminal) factor"""
        factors = pd.DataFrame([Factor(id='taps')])
        branchterminals = _branchterminal_frame(
            pd.DataFrame([
                Branch(id='br', id_of_node_A='n_0', id_of_node_B='n_1')]),
            pd.Series(['n_0'], name='id_of_node', dtype=str))
        terminallinks = _terminallink_frame(
            pd.DataFrame([
                Terminallink(branchid='br', nodeid='n_0', id='taps')]))
        factordefs = make_factordefs(
            factors,
            terminallinks,
            self.no_injectionlinks,
            branchterminals)
        self.assertEqual(
            len(factordefs.gen_factor_data),
            1,
            "make_factordefs shall return one generic factor")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "make_factordefs shall not return return "
            "any link: injection -> generic factor")
        self.assertEqual(
            len(factordefs.gen_termfactor),
            1,
            "make_factordefs shall return "
            "one link: terminal -> generic factor")
        self.assertTrue(
            all(factordefs.get_groups([idx]).empty
                for idx in range(5)),
            "make_factordefs shall not return any factor for any step")
        self.assertTrue(
            all(factordefs.get_injfactorgroups([idx]).empty
                for idx in range(5)),
            "make_factordefs shall not return any link "
            "from injection to factor for any step")

if __name__ == '__main__':
    unittest.main()
