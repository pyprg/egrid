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
from egrid.model import (
    _get_pfc_nodes, _prepare_branches, _get_branch_terminals, _add_bg)
from egrid._types import (
    FACTORS, TERMINALLINKS, INJLINKS, BRANCHES, SLACKNODES)
from egrid.factors import make_factordefs

class Make_factordefs(unittest.TestCase):

    def test_empty(self):
        terminallinks = (
            TERMINALLINKS.copy()
            .set_index(['step', 'branchid', 'nodeid']))
        terminallinks.index.names=['step', 'id_of_branch', 'id_of_node']
        injectionlinks = (
            INJLINKS.copy()
            .set_index(['step', 'injid', 'part']))
        injectionlinks.index.names = ['step', 'id_of_injection', 'part']
        branches_ = BRANCHES.copy()
        branches_['is_bridge'] = False
        pfc_slack_count, node_count, pfc_nodes = _get_pfc_nodes(
            SLACKNODES.id_of_node, branches_)
        branches = _prepare_branches(branches_, pfc_nodes)
        branchterminals = _get_branch_terminals(_add_bg(branches))
        factordefs = make_factordefs(
            FACTORS.copy(), terminallinks, injectionlinks, branchterminals)
        self.assertTrue(
            factordefs.gen_factor_data.empty,
            "no generic factors")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "no links from generic factors to links")
        self.assertTrue(
            factordefs.gen_termfactor.empty,
            "no links from generic factors to terminals")
        self.assertTrue(
            all(factordefs.get_groups([idx]).empty
                for idx in range(-1, 5)),
            "no factors for any step")
        self.assertTrue(
            all(factordefs.get_injfactorgroups([idx]).empty
                for idx in range(-1, 5)),
            "no links from injections to factor for any step")

if __name__ == '__main__':
    unittest.main()
