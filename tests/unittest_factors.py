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
import context
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from itertools import repeat
from egrid.model import (
    _get_pfc_nodes, _prepare_branches, _get_branch_terminals, _add_bg,
    model_from_frames)
from egrid.builder import (
    Factor, Injectionlink, Terminallink, Branch, make_data_frames)
from egrid._types import (
    FACTORS, TERMINALLINKS, INJLINKS, BRANCHES, Injection, DEFAULT_FACTOR_ID,
    Slacknode, Defk, Deft, Klink, Tlink)
from egrid.factors import (
    make_factordefs, _get_scaling_factor_data, make_factor_meta,
    _get_taps_factor_data, get_factors)

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
    count_of_branches = len(branches)
    branches2 = _prepare_branches(branches_, pfc_nodes, count_of_branches)
    return _get_branch_terminals(_add_bg(branches2), count_of_branches)

class Make_factordefs(unittest.TestCase):

    def test_no_data(self):
        model = model_from_frames(
            make_data_frames())
        factordefs = get_factors(model)
        self.assertTrue(
            factordefs.gen_factordata.empty, 'no generic factors')
        self.assertTrue(
            factordefs.gen_injfactor.empty, 'no generic injection factors')
        self.assertTrue(
            factordefs.terminalfactors.empty, 'no generic terminal factors')

    def test_generic_injection_factor(self):
        """basic test with one generic injection factor"""
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Branch(
                    id='branch',
                    id_of_node_A='n_0',
                    id_of_node_B='n_1'),
                Injection('consumer', 'n_1'),
                # scaling, define scaling factors
                Defk(id='kp', step=-1),
                # link scaling factors to active power of consumer
                #   factor for each step (generic, step=-1)
                Klink(
                    id_of_injection='consumer',
                    part='p',
                    id_of_factor='kp',
                    step=-1)]))
        factordefs = get_factors(model)
        self.assertEqual(
            dict(
                zip(factordefs.gen_factordata.columns,
                    factordefs.gen_factordata.iloc[0].to_numpy())),
            {'step': -1, 'type': 'var', 'id_of_source': 'kp', 'value': 1.0,
              'min': -np.inf, 'max': np.inf, 'is_discrete': False, 'm': 1.0,
              'n': 0.0, 'cost': 0., 'index_of_symbol': 0})
        self.assertEqual(
            factordefs.gen_factordata.index[0],
            'kp',
            "generic factor has name 'kp'")
        assert_array_equal(
            factordefs.gen_injfactor.to_numpy(),
            np.array([[-1, 'kp']], dtype=object),
            err_msg="expected one generic factor (-1) with id 'kp'")
        assert_array_equal(
            factordefs.gen_injfactor.index.to_numpy()[0],
            ('consumer', 'p'),
            err_msg="expected index is ('consumer', 'p')")
        assert_array_equal(
            factordefs.terminalfactors,
            np.zeros((0,7), dtype=object),
            err_msg="no taps (terminal) factor"),
        self.assertEqual(
            len(factordefs.get_groups([-1])),
            1,
            "one generic factor")
        self.assertEqual(
            len(factordefs.get_injfactorgroups([-1])),
            1,
            "one generic injection_factor relation")

    def test_taps_injection_factor(self):
        """basic test with one generic taps (terminal) factor"""
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Branch(
                    id='branch',
                    id_of_node_A='n_0',
                    id_of_node_B='n_1',
                    y_lo=1e4),
                Injection('consumer', 'n_1'),
                # scaling, define scaling factors
                Deft(
                    id='taps', value=0., type='const', is_discrete=True,
                    m=-0.00625, n=1., step=-1),
                # link scaling factors to active power of consumer
                #   factor for each step (generic, step=-1)
                Tlink(
                    id_of_node='n_0',
                    id_of_branch='branch',
                    id_of_factor='taps',
                    step=-1)]))
        factordefs = get_factors(model)
        self.assertEqual(
            dict(
                zip(factordefs.gen_factordata.columns,
                    factordefs.gen_factordata.iloc[0].to_numpy())),
            {'step': -1, 'type': 'const', 'id_of_source': 'taps', 'value': 0.,
              'min': -16, 'max': 16, 'is_discrete': True, 'm': -0.00625,
              'n': 1.0, 'cost': 0., 'index_of_symbol': 0})
        self.assertEqual(
            factordefs.gen_factordata.index[0],
            'taps',
            "generic factor has name 'taps'")
        assert_array_equal(
            factordefs.gen_injfactor,
            np.zeros((0,2), dtype=object),
            err_msg="expected no generic injection_factor relation")
        assert_array_equal(
            factordefs.terminalfactors.to_numpy(),
            np.array(
                [['taps', 0, 1, 0.0, -0.00625, 1.0, 0]],
                dtype=object),
            err_msg="expected taps (terminal) factor "
            "['taps', 0, 1, 0.0, -0.00625, 1.0, 0]"),
        assert_array_equal(
            factordefs.terminalfactors.id.to_numpy()[0],
            ['taps'],
            err_msg="expected id is 'taps'")
        self.assertEqual(
            len(factordefs.get_groups([-1])),
            1,
            "one generic factor")

class Get_taps_factor_data(unittest.TestCase):

    def test_empty_model(self):
        model = model_from_frames(make_data_frames())
        model_factors = get_factors(model)
        factors, termfactor_crossref = _get_taps_factor_data(
            model_factors, [0, 1])
        self.assertTrue(factors.empty)
        self.assertTrue(termfactor_crossref.empty)

    def test_generic_specific_factor(self):
        """
        one generic injection factor 'kp',
        one generic terminal factor'taps',
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Branch(
                    id='branch',
                    id_of_node_A='n_0',
                    id_of_node_B='n_1',
                    y_lo=1e4),
                Injection('consumer', 'n_1'),
                Injection('consumer2', 'n_1'),
                # scaling, define scaling factors
                Defk(id='kp', step=-1),
                Defk(id='kq', step=0),
                # link scaling factors to active and reactive power of consumer
                #   factor for each step (generic, step=-1)
                Klink(
                    id_of_injection='consumer',
                    id_of_factor='kp',
                    part='p',
                    step=-1),
                #   factor for step 0 (specific, step=0)
                Klink(
                    id_of_injection='consumer',
                    id_of_factor='kq',
                    part='q',
                    step=0),
                # tap factor, for each step (generic, step=-1)
                Deft(id='taps', is_discrete=True, step=-1),
                # tap factor, for step 1
                Deft(id='taps', type='const', is_discrete=True, step=1),
                # link generic tap factor to terminal
                Tlink(
                    id_of_node='n_0',
                    id_of_branch='branch',
                    id_of_factor='taps',
                    step=-1),
                Tlink(
                    id_of_node='n_1',
                    id_of_branch='branch',
                    id_of_factor='taps',
                    step=-1),
                # # link step specific tap factor to terminal
                # grid.Link(
                #     objid='branch',
                #     id='taps',
                #     nodeid='n_0',
                #     cls=grid.Terminallink,
                #     step=1)
                ]))
        model_factors = get_factors(model)
        assert_array_equal(
            model_factors.gen_factordata.index,
            ['kp', 'taps'],
            err_msg="IDs of generic factors shall be ['kp', 'taps']")
        assert_array_equal(
            model_factors.gen_factordata.index_of_symbol,
            [0, 1],
            err_msg="indices of generic factor symbols shall be [0, 1]")
        factors, termfactor_crossref = _get_taps_factor_data(
            model_factors, [0,1])
        print('add tests')

class Get_scaling_factor_data(unittest.TestCase):

    def test_no_data(self):
        """well, """
        model = model_from_frames(make_data_frames())
        model_factors = get_factors(model)
        factors, injfactor_crossref = _get_scaling_factor_data(
            model_factors, model.injections, [0, 1],
            repeat(len(model_factors.gen_factordata)))
        self.assertTrue(factors.empty)
        self.assertTrue(injfactor_crossref.empty)

    def test_default_scaling_factors(self):
        """
        one generic injection factor 'kp',
        one generic terminal factor'taps',
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Branch(
                    id='branch',
                    id_of_node_A='n_0',
                    id_of_node_B='n_1',
                    y_lo=1e4),
                Injection('consumer', 'n_1'),
                Injection('consumer2', 'n_1'),
                # scaling, define scaling factors
                Defk(id='kp', step=-1),
                Defk(id='kq', step=0),
                # link scaling factors to active and reactive power of consumer
                #   factor for each step (generic, step=-1)
                Klink(
                    id_of_injection='consumer',
                    id_of_factor='kp',
                    part='p',
                    step=-1),
                #   factor for step 0 (specific, step=0)
                Klink(
                    id_of_injection='consumer',
                    id_of_factor='kq',
                    part='q',
                    step=0),
                # tap factor, for each step (generic, step=-1)
                Deft(id='taps', is_discrete=True, step=-1),
                # link scaling factors to active and reactive power of consumer
                Tlink(
                    id_of_branch='branch',
                    id_of_factor='taps',
                    id_of_node='n_0',
                    step=-1)]))
        model_factors = get_factors(model)
        assert_array_equal(
            model_factors.gen_factordata.index,
            ['kp', 'taps'],
            err_msg="IDs of generic factors shall be ['kp', 'taps']")
        assert_array_equal(
            model_factors.gen_factordata.index_of_symbol,
            [0, 1],
            err_msg="indices of generic factor symbols shall be [0, 1]")
        factors, crossref = _get_scaling_factor_data(
            model_factors, model.injections, [0, 1],
            repeat(len(model_factors.gen_factordata)))
        assert_array_equal(
            factors.loc[0].index.get_level_values('id').sort_values(),
            ['_default_', 'kp', 'kq'],
            err_msg="factor ids of step 0 are ['_default_', 'kp', 'kq']")
        assert_array_equal(
            factors.loc[0].index_of_symbol,
            [2, 0, 3],
            err_msg="indices of symbols for step 0 are [2, 0, 3]")
        assert_array_equal(
            factors.loc[0].index_of_source,
            [-1, -1, -1],
            err_msg="indices of symbols for step 0 are [-1, -1, -1]")
        assert_array_equal(
            factors.loc[1].index.get_level_values('id').sort_values(),
            ['_default_', 'kp'],
            err_msg="factor ids of step 1 are ['_default_', 'kp']")
        assert_array_equal(
            factors.loc[1].index_of_symbol,
            [2, 0],
            err_msg="indices of symbols for step 1 are [2, 0]")
        assert_array_equal(
            factors.loc[1].index_of_source,
            [2, 0],
            err_msg="indices of symbols for step 1 are [2, 0]")

    def test_missing_generic_links(self):
        """
        no generic injection factor 'kp' as link is missing,
        one generic terminal factor'taps',
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Branch(
                    id='branch',
                    id_of_node_A='n_0',
                    id_of_node_B='n_1',
                    y_lo=1e4),
                Injection('consumer', 'n_1'),
                Injection('consumer2', 'n_1'),
                # scaling, define scaling factors
                Defk(id='kp', step=-1),
                Defk(id='kq', step=0),
                # link scaling factors to active and reactive power of consumer
                #   factor for each step (generic, step=-1)
                # grid.Klink(id_of_injection='consumer', id_of_factor='kp', part='p', step=-1),
                #   factor for step 0 (specific, step=0)
                Klink(
                    id_of_injection='consumer',
                    part='q',
                    id_of_factor='kq',
                    step=0),
                # tap factor, for each step (generic, step=-1)
                Deft(id='taps', is_discrete=True, step=-1),
                # link scaling factors to active and reactive power of consumer
                Tlink(
                    id_of_node='n_0',
                    id_of_branch='branch',
                    id_of_factor='taps',
                    step=-1)]))
        model_factors = get_factors(model)
        assert_array_equal(
            model_factors.gen_factordata.index,
            ['taps'],
            err_msg="IDs of generic factors shall be ['taps']")
        assert_array_equal(
            model_factors.gen_factordata.index_of_symbol,
            [0],
            err_msg="indices of generic factor symbols shall be [0]")
        factors, crossref = _get_scaling_factor_data(
            model_factors, model.injections, [0, 1],
            repeat(len(model_factors.gen_factordata)))
        assert_array_equal(
            factors.loc[0].index.get_level_values('id').sort_values(),
            ['_default_', 'kq'],
            err_msg="factor ids of step 0 are ['_default_', 'kq']")
        assert_array_equal(
            factors.loc[0].index_of_symbol,
            [1, 2],
            err_msg="indices of symbols for step 0 are [1, 2]")
        assert_array_equal(
            factors.loc[0].index_of_source,
            [-1, -1],
            err_msg="indices of symbols for step 0 are [-1, -1]")
        assert_array_equal(
            factors.loc[1].index.get_level_values('id').sort_values(),
            ['_default_'],
            err_msg="factor ids of step 1 are ['_default_']")
        assert_array_equal(
            factors.loc[1].index_of_symbol,
            [1],
            err_msg="indices of symbols for step 1 are [1]")
        assert_array_equal(
            factors.loc[1].index_of_source,
            [1],
            err_msg="indices of symbols for step 1 are [1]")

class Make_factordefs2(unittest.TestCase):

    # prepare empty pandas.DataFrame instances
    no_terminallinks = _terminallink_frame(TERMINALLINKS.copy())
    no_injectionlinks = _injectionlink_frame(INJLINKS.copy())
    no_branchterminals = _branchterminal_frame(
        BRANCHES.copy(), pd.Series([], name='id_of_node', dtype=str))
    no_factors = FACTORS.copy()

    def test_empty(self):
        """without data, no factors at all"""
        factordefs = make_factordefs(
            self.no_factors,
            self.no_terminallinks,
            self.no_injectionlinks,
            self.no_branchterminals)
        self.assertTrue(
            factordefs.gen_factordata.empty,
            "no generic factors")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "no links from injections to generic factors")
        self.assertTrue(
            factordefs.terminalfactors.empty,
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
        """one generic scaling factor (part 'p')"""
        factors = pd.DataFrame([Factor(id='kp')])
        injectionlinks = _injectionlink_frame(
            pd.DataFrame([Injectionlink(injid='i_0', part='p', id='kp')]))
        factordefs = make_factordefs(
            factors,
            self.no_terminallinks,
            injectionlinks,
            self.no_branchterminals)
        self.assertEqual(
            len(factordefs.gen_factordata),
            1,
            "make_factordefs shall return one generic factor")
        self.assertEqual(
            len(factordefs.gen_injfactor),
            1,
            "make_factordefs shall return "
            "one link: injection -> generic factor")
        self.assertTrue(
            factordefs.terminalfactors.empty,
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

    def test_generic_scaling_factor2(self):
        """3 generic scaling factors"""
        factors = pd.DataFrame(
            [Factor(id='kp0'),
              Factor(id='kq0'),
              Factor(id='kp1')])
        injectionlinks = _injectionlink_frame(
            pd.DataFrame(
                [Injectionlink(injid='i_0', part='p', id='kp0'),
                  Injectionlink(injid='i_0', part='q', id='kq0'),
                  Injectionlink(injid='i_1', part='p', id='kp1')]))
        factordefs = make_factordefs(
            factors,
            self.no_terminallinks,
            injectionlinks,
            self.no_branchterminals)
        self.assertEqual(
            len(factordefs.gen_factordata),
            3,
            "make_factordefs shall return 3 generic factors")
        self.assertEqual(
            set(factordefs.gen_factordata.index),
            {'kp1', 'kp0', 'kq0'},
            "make_factordefs shall return factors 'kp1', 'kp0', 'kq0'")
        self.assertEqual(
            len(factordefs.gen_injfactor),
            3,
            "make_factordefs shall return "
            "3 links: injection -> generic factor")
        self.assertTrue(
            factordefs.terminalfactors.empty,
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
        """one scaling factor (part 'q') for step 0"""
        factors = pd.DataFrame([Factor(id='kq', step=0)])
        injectionlinks = _injectionlink_frame(
            pd.DataFrame(
                [Injectionlink(injid='i_0', part='q', id='kq', step=0)]))
        factordefs = make_factordefs(
            factors,
            self.no_terminallinks,
            injectionlinks,
            self.no_branchterminals)
        self.assertTrue(
            factordefs.gen_factordata.empty,
            "make_factordefs shall not return any generic factor")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "no links from injections to generic factors")
        self.assertTrue(
            factordefs.terminalfactors.empty,
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

    def test_scaling_factor_step02(self):
        """one scaling factor (part 'q') for step 0, 2"""
        factors = pd.DataFrame(
            [Factor(id='kq', step=0),
              Factor(id='kq', step=2)])
        injectionlinks = _injectionlink_frame(
            pd.DataFrame(
                [Injectionlink(injid='i_0', part='q', id='kq', step=0),
                  Injectionlink(injid='i_0', part='q', id='kq', step=2)]))
        factordefs = make_factordefs(
            factors,
            self.no_terminallinks,
            injectionlinks,
            self.no_branchterminals)
        self.assertTrue(
            factordefs.gen_factordata.empty,
            "make_factordefs shall not return any generic factor")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "no links from injections to generic factors")
        self.assertTrue(
            factordefs.terminalfactors.empty,
            "make_factordefs shall not return "
            "any link from terminal to generic factor")
        assert_array_equal(
            [len(factordefs.get_groups([idx])) for idx in range(-1, 5)],
            [0, 1, 0, 1, 0, 0],
            err_msg="make_factordefs shall return one factor for steps 0, 2")
        assert_array_equal(
            [len(factordefs.get_injfactorgroups([idx]))
              for idx in range(-1, 5)],
            [0, 1, 0, 1, 0, 0],
            err_msg="make_factordefs shall return one link injection->factor "
            "for steps 0, 2")

    def test_scaling_factor_step01(self):
        """two scaling factors (part 'pq') for step 0, one (part 'q')
        for step 1"""
        factors = pd.DataFrame(
            [Factor(id='kp', step=0),
              Factor(id='kq', step=0),
              Factor(id='kq', step=1)])
        injectionlinks = _injectionlink_frame(
            pd.DataFrame(
                [Injectionlink(injid='i_0', part='p', id='kp', step=0),
                  Injectionlink(injid='i_0', part='q', id='kq', step=0),
                  Injectionlink(injid='i_0', part='q', id='kq', step=1)]))
        factordefs = make_factordefs(
            factors,
            self.no_terminallinks,
            injectionlinks,
            self.no_branchterminals)
        self.assertTrue(
            factordefs.gen_factordata.empty,
            "make_factordefs shall not return any generic factor")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "no links from injections to generic factors")
        self.assertTrue(
            factordefs.terminalfactors.empty,
            "make_factordefs shall not return "
            "any link from terminal to generic factor")
        assert_array_equal(
            [len(factordefs.get_groups([idx])) for idx in range(-1, 5)],
            [0, 2, 1, 0, 0, 0],
            err_msg="make_factordefs shall return two factors for steps 0"
            "and one for step 1")
        self.assertEqual(
            set(factordefs.get_groups([0]).index.levels[1]),
            {'kq', 'kp'},
            "make_factordefs shall return factors 'kp' and 'kq' for step 0")
        self.assertEqual(
            set(factordefs.get_groups([1]).index.levels[1]),
            {'kq'},
            "make_factordefs shall return factor 'kq' for step 1")
        assert_array_equal(
            [len(factordefs.get_injfactorgroups([idx]))
              for idx in range(-1, 5)],
            [0, 2, 1, 0, 0, 0],
            err_msg="make_factordefs shall return two links injection->factor "
            "for steps 0 and one for step 1")

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
            len(factordefs.gen_factordata),
            1,
            "make_factordefs shall return one generic factor")
        self.assertTrue(
            factordefs.gen_injfactor.empty,
            "make_factordefs shall not return "
            "any link: injection -> generic factor")
        self.assertEqual(
            len(factordefs.terminalfactors),
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

class Get_scaling_factor_data2(unittest.TestCase):

    def test_no_data(self):
        """'get_scaling_factor_data' processes empty input"""
        model = model_from_frames(make_data_frames())
        model_factors = get_factors(model)
        factors, injection_factor = _get_scaling_factor_data(
            model_factors, model.injections, [2, 3], None)
        self.assertTrue(
            factors.empty,
            "get_scaling_factor_data returns no data for factors")
        self.assertTrue(
            injection_factor.empty,
            "get_scaling_factor_data returns no data for association "
            "injection_factor")

    def test_default_scaling_factors(self):
        """'get_scaling_factor_data' creates default scaling factors
        if factors are not given explicitely"""
        model = model_from_frames(
            make_data_frames([Injection(id='injid0', id_of_node='n_0')]))
        index_of_step = 3
        steps = [index_of_step-1, index_of_step]
        model_factors = get_factors(model)
        factors, injection_factor = _get_scaling_factor_data(
            model_factors, model.injections, steps, None)
        assert_array_equal(
            [idx[0] for idx in factors.index],
            steps,
            err_msg="one factor per step")
        self.assertTrue(
            all(idx[1]=='const' for idx in factors.index),
            "all factors are const parameters")
        self.assertTrue(
            all(idx[2]==DEFAULT_FACTOR_ID for idx in factors.index),
            f"all factors are '{DEFAULT_FACTOR_ID}'")

    def test_step1_without_scaling_factordef(self):
        """'get_scaling_factor_data' creates default scaling factors
        if factors are not given explicitely"""
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Injection('consumer', 'n_0'),
                # scaling, define scaling factors
                Defk(id=('kp', 'kq'), step=0),
                # link scaling factors to active and reactive power of consumer
                Klink(
                    id_of_injection='consumer',
                    part='pq',
                    id_of_factor=('kp', 'kq'),
                    step=0)]))
        index_of_step = 1
        steps = [index_of_step-1, index_of_step]
        model_factors = get_factors(model)
        factors, injection_factor = _get_scaling_factor_data(
            model_factors, model.injections, steps, None)
        self.assertEqual(
            factors.loc[0].shape[0],
            2,
            "two factors for step 0")
        self.assertTrue(
            all(factors.loc[0].reset_index().type == 'var'),
            "all factors of step 0 are decision variables ('var')")
        self.assertEqual(
            factors.loc[1].shape[0],
            1,
            "one factors for step 1")
        self.assertTrue(
            all(factors.loc[1].reset_index().type == 'const'),
            "all factors of step 1 are parameters ('const')")
        self.assertTrue(
            all(factors.loc[1].reset_index().id == DEFAULT_FACTOR_ID),
            f"id of factor for step 1 is '{DEFAULT_FACTOR_ID}'")
        self.assertTrue(
            all(factors.index_of_source == -1),
            "there are no source factors")

    def test_generic_scaling_factor(self):
        """'get_scaling_factor_data' creates default factors if factors are not
        given explicitely"""
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Injection('consumer', 'n_0'),
                # scaling, define scaling factors
                Defk(id=('kp', 'kq'), step=-1),
                # link scaling factors to active and reactive power of consumer
                Klink(
                    id_of_injection='consumer',
                    part='pq',
                    id_of_factor=('kp', 'kq'),
                    step=-1)]))
        index_of_step = 1
        steps = [index_of_step-1, index_of_step]
        model_factors = get_factors(model)
        factors, injection_factor = _get_scaling_factor_data(
            model_factors, model.injections, steps, None)
        factors_step_0 = factors.loc[0]
        self.assertEqual(
            len(factors_step_0),
            2,
            "two factors for step 0")
        factors_step_1 = factors.loc[1]
        self.assertEqual(
            len(factors_step_1),
            2,
            "two factors for step 1")

    def test_scaling_factor_with_terminallink(self):
        """'get_scaling_factor_data' creates factors for injections
        if linked with Injectionlink only"""
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Injection('consumer', 'n_0'),
                # scaling, define scaling factors
                Defk(id=('kp', 'kq'), step=-1),
                # link scaling factors to active and reactive power of consumer
                Tlink(
                    id_of_branch='consumer',
                    id_of_node='n_0',
                    id_of_factor=('kp', 'kq'),
                    step=-1)]))
        index_of_step = 0
        steps = [index_of_step]
        model_factors = get_factors(model)
        factors, injection_factor = _get_scaling_factor_data(
            model_factors, model.injections, steps, None)
        self.assertEqual(
            len(factors),
            1,
            "one factor")
        self.assertEqual(
            factors.index[0][2],
            DEFAULT_FACTOR_ID,
            "'get_scaling_factor_data' creates a scaling factor with ID "
            f"{DEFAULT_FACTOR_ID} as link is a Terminallink")

class Get_taps_factor_data2(unittest.TestCase):

    def test_terminal_factor(self):
        """"""
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Branch(
                    id='branch',
                    id_of_node_A='n_0',
                    id_of_node_B='n_1'),
                Injection('injection', 'n_1'),
                # scaling, define scaling factors
                Deft(id='taps', step=-1),
                # link scaling factors to active and reactive power of consumer
                Tlink(
                    id_of_node='n_0',
                    id_of_branch='branch',
                    id_of_factor='taps',
                    step=-1)]))
        index_of_step = 1
        steps = [index_of_step-1, index_of_step]
        model_factors = get_factors(model)
        factors, injection_factor = _get_scaling_factor_data(
            model_factors, model.injections, steps, start=[3, 5])
        self.assertEqual(
            len(factors.loc[0]),
            1,
            "one factor for step 0")
        self.assertEqual(
            factors.loc[0].index_of_symbol.iloc[0],
            3,
            "step-0 symbol has index 3")
        self.assertEqual(
            len(factors.loc[1]),
            1,
            "one factor for step 1")
        self.assertEqual(
            factors.loc[1].index_of_symbol.iloc[0],
            5,
            "step-1 symbol has index 5")

class Make_factor_meta(unittest.TestCase):

    def test_no_data(self):
        model = model_from_frames(make_data_frames())
        factors = get_factors(model)
        factor_meta = make_factor_meta(
            model, factors, 1, np.zeros((0,1), dtype=float))
        self.assertEqual(
            factor_meta.id_of_step_symbol.shape,
            (0,),
            "no symbols specific for step 1")
        self.assertEqual(
            factor_meta.index_of_kpq_symbol.shape,
            (0,2),
            "no scaling factors")
        self.assertEqual(
            factor_meta.index_of_var_symbol.shape,
            (0,),
            "no decision variables")
        self.assertEqual(
            factor_meta.index_of_const_symbol.shape,
            (0,),
            "no values for decision variables")
        self.assertEqual(
            factor_meta.values_of_vars.shape,
            (0,),
            "no paramters")
        self.assertEqual(
            factor_meta.var_min.shape,
            (0,),
            "no minimum values for decision variables")
        self.assertEqual(
            factor_meta.var_max.shape,
            (0,),
            "no maximum values for decision variables")
        self.assertEqual(
            factor_meta.values_of_consts.shape,
            (0,),
            "no values for paramters")
        assert_array_equal(
            factor_meta.var_const_to_factor,
            [],
            err_msg="no var_const crossreference")
        assert_array_equal(
            factor_meta.var_const_to_kp,
            [],
            err_msg="no active power scaling factor crossreference")
        assert_array_equal(
            factor_meta.var_const_to_kq,
            [],
            err_msg="no reactive power scaling factor crossreference")
        assert_array_equal(
            factor_meta.var_const_to_ftaps,
            [],
            err_msg="no taps factor crossreference")

    def test_generic_specific_factor(self):
        """
        one generic injection factors 'kp'
        generic terminal factor'tap_', 'taps2'
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = model_from_frames(
            make_data_frames([
                Slacknode('n_0'),
                Branch(
                    id='branch', id_of_node_A='n_0', id_of_node_B='n_1',
                    y_lo=1e4),
                Branch(
                    id='branch2', id_of_node_A='n_1', id_of_node_B='n_2',
                    y_lo=1e4),
                Injection('consumer', 'n_1'), # has kp and kq
                Injection('consumer2', 'n_1'),# has _default_ factor for P, Q
                # scaling, define scaling factors
                Defk(id='kp', step=-1),
                Defk(id='kq', step=0),
                # link scaling factors to active and reactive power of consumer
                #   factor for each step (generic, step=-1)
                Klink(
                    id_of_injection='consumer',
                    part='p',
                    id_of_factor='kp',
                    step=-1),
                #   factor for step 0 (specific, step=0)
                Klink(
                    id_of_injection='consumer',
                    part='q',
                    id_of_factor='kq',
                    step=0),
                # tap factor, for each step (generic, step=-1)
                Deft(id='tap_', is_discrete=True, step=-1),
                Deft(id='tap2', is_discrete=True, step=-1),
                # link the same generic tap factor to both terminals
                Tlink(
                    id_of_node='n_0',
                    id_of_branch='branch',
                    id_of_factor='tap_',
                    step=-1),
                Tlink(
                    id_of_node='n_1',
                    id_of_branch='branch',
                    id_of_factor='tap2',
                    step=-1)]))
        model_factors = get_factors(model)
        assert_array_equal(
            model_factors.gen_factordata.index,
            ['kp', 'tap2', 'tap_'],
            err_msg="IDs of generic factors shall be ['kp', 'tap2', 'tap_']")
        assert_array_equal(
            model_factors.gen_factordata.index_of_symbol,
            [0, 1, 2],
            err_msg="indices of generic factor symbols shall be [0, 1, 2]")
        fm = make_factor_meta(
            model, model_factors, 0, np.zeros((0,1), dtype=float))
        # prepare generic factors
        generic_factors = (
            model_factors
            .gen_factordata
            .reset_index()
            .sort_values(by='index_of_symbol'))
        # use 'factor_ids' as a substitute for symbols during this test
        # symbols of generic factors can be created before optimization and
        #   can be made available for each step,
        #   first index of generic factor is 0
        # step-specific can be created before a specific step,
        #     symbols start with index len(generic_factors)
        # elements of 'factor_ids' are in reference order with increasing
        #   index, starting with 0, independent of type 'var'|'const'
        # method for arranging symbols in reference order
        #   'generic_factors.id' and 'fm.id_of_step_symbol' are in correct
        #   order having an increasing index without gap, all methodes using
        #   index-sequences 'index_of_kpq_symbol', 'index_of_var_symbol'
        #   and 'index_of_const_symbol' for ordering of values rely on this
        #   order
        factor_ids = (
            pd.concat([generic_factors.id, fm.id_of_step_symbol]).to_numpy())
        # method for arranging kp and kq for each injection
        kpq=pd.DataFrame(
            {'kp': factor_ids[fm.index_of_kpq_symbol[:,0]],
              'kq': factor_ids[fm.index_of_kpq_symbol[:,1]]}).to_numpy()
        # method for extracting all decision variable symbols
        vars=factor_ids[fm.index_of_var_symbol]
        # method for extracting all parameter symbols
        consts=factor_ids[fm.index_of_const_symbol]
        # vars are ordered in the way the solver returns the values
        #  use var_const as substitute for values returned by a solver during
        #  this test
        var_const = np.concatenate([vars, consts])
        # extract scaling factors for active power from solver results
        kp = var_const[fm.var_const_to_kp]
        # extract scaling factors for reactive power from solver results
        kq = var_const[fm.var_const_to_kq]
        # extracting terminal factors from solver result values
        ftaps = var_const[fm.var_const_to_ftaps]
        # reordering of solver result values, this is necessary before passing
        #   as initial values for next step
        factor_ids2 = var_const[fm.var_const_to_factor]
        # method for creating a terminal-factor for each terminal
        #   tfactors = np.ones((len(model.branchterminals)), dtype=float)
        #   tfactors[tfdata.index_of_terminal] = ftaps
        # for test we use numpy.empty with dtype='<U4'
        terminal_factors = np.empty((len(model.branchterminals),), dtype='<U4')
        idxs = model_factors.terminalfactors.index_of_terminal
        terminal_factors[idxs] = ftaps
        # check
        assert_array_equal(
            kpq, [['kp', 'kq'], [DEFAULT_FACTOR_ID, DEFAULT_FACTOR_ID]])
        assert_array_equal(vars, ['kp', 'kq'])
        assert_array_equal(consts, ['tap2', 'tap_', DEFAULT_FACTOR_ID])
        assert_array_equal(kp, ['kp', DEFAULT_FACTOR_ID])
        assert_array_equal(kq, ['kq', DEFAULT_FACTOR_ID])
        assert_array_equal(ftaps, ['tap_', 'tap2'])
        assert_array_equal(
            factor_ids,
            factor_ids2,
            err_msg="results shall be ordered according to concatenation of "
            "generic factors and step-specific factors")
        assert_array_equal(terminal_factors, ['tap_', '', 'tap2', ''])

if __name__ == '__main__':
    unittest.main()
