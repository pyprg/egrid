# -*- coding: utf-8 -*-
"""
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

Created on Sun Dec 10 16:39:23 2023

@author: live
"""

import unittest
import context
import pandas as pd
from egrid import _make_model
from egrid._types import (
    FACTORS, TERMINALLINKS, INJLINKS, BRANCHES, Injection, DEFAULT_FACTOR_ID,
    Branch, Injection, Slacknode, Defk, Deft, Klink, Tlink, PValue, QValue,
    Output, Factor)
from egrid.factorgen import (
    get_parts_of_injections, get_pq_subgraphs, make_scaling_factors)




"""
                        P=-5
                        Q=-3
n0<---br00--->n1<---br01--->n2<---br02--->n3<---br03--->n4<---br04--->n5
                            |
                            |
                            |
                            n2<---br02_2--->n3_2
                            |
                            |
                            +-->inj02



"""
MODEL = _make_model([
    Slacknode(id_of_node='n0'),

    Branch(
        id='br00', id_of_node_A='n0', id_of_node_B='n1', y_lo=1e3+1e3j),
    Branch(
        id='br01', id_of_node_A='n1', id_of_node_B='n2', y_lo=1e3+1e3j),
    Branch(
        id='br02', id_of_node_A='n2', id_of_node_B='n3', y_lo=1e3+1e3j),
    Branch(
        id='br02_2', id_of_node_A='n2', id_of_node_B='n3_2',
        y_lo=1e3+1e3j),

    Injection(id='inj03_0', id_of_node='n3'),
    Injection(id='inj04_0', id_of_node='n4', P10=2),
    Injection(id='inj02', id_of_node='n2', P10=3, Q10=13),
    Injection(id='inj02_3', id_of_node='n5', P10=5, Q10=17),
    Injection(id='inj03_4', id_of_node='n3_2', P10=7, Q10=19),
    Injection(id='inj03_5', id_of_node='n3_2', P10=11, Q10=23),

    Branch(
        id='br03', id_of_node_A='n3', id_of_node_B='n4'),
    Branch(
        id='br04', id_of_node_A='n4', id_of_node_B='n5', y_lo=1e3+1e3j),

    PValue(id_of_batch='n2_br02', P=5),
    QValue(id_of_batch='n2_br02', Q=3),
    Output(
        id_of_batch='n2_br02', id_of_node='n2', id_of_device='br02'),
    Output(
        id_of_batch='n2_br02', id_of_node='n2', id_of_device='br02_2'),
    Output(id_of_batch='n2_br02', id_of_device='inj02'),

    Factor(id='kp'),
    Factor(id='kp_const', type='const'),
    Factor(id='kp0', step=0),
    Factor(id='kq'),
    Klink(id_of_factor='kp_const', id_of_injection='inj03_5', part='p'),
    Klink(id_of_factor='kp', id_of_injection='inj02', part='p'),
    Klink(id_of_factor='kp', id_of_injection='inj02_3', part='p', step=0),
    Klink(id_of_factor='kq', id_of_injection='inj02_3', part='q', step=0),
    Klink(id_of_factor='kp', id_of_injection='inj03_4', part='p'),
    Klink(id_of_factor='kp0', id_of_injection='inj03_4', part='p', step=0),
    # reference to not existing factor
    Klink(
        id_of_factor='kq0', id_of_injection='inj03_4', part='q', step=0)])

class Get_parts_of_injections(unittest.TestCase):

    def test_get_parts_of_injections(self):
        parts, factors = get_parts_of_injections(MODEL, step=0, PQlimit=.01)
        self.assertEqual(
            parts.index.get_level_values(0).to_list(),
            ['inj03_0', 'inj03_0', 'inj04_0', 'inj04_0', 'inj02', 'inj02',
             'inj02_3', 'inj02_3', 'inj03_4', 'inj03_4', 'inj03_5', 'inj03_5'])
        self.assertEqual(
            parts.index.get_level_values(1).to_list(),
            ['P', 'Q', 'P', 'Q', 'P', 'Q', 'P', 'Q', 'P', 'Q', 'P', 'Q'])
        self.assertEqual(
            parts.is_scalable.to_list(),
            [False, False, True, False, True, True, True, True, True, True,
             False, True])
        self.assertEqual(
            parts.var_type.to_list(),
            ['var', 'var', 'var', 'var', 'var', 'var', 'var', 'var',
              'var', 'var', 'const', 'var'])

class Get_initial_scaling_factors(unittest.TestCase):

    def test_get_initial_scaling_factors(self):
        # initial values
        ini_values = (
            .7 *
            MODEL.injections
            .loc[:,['id','P10','Q10']]
            .rename(columns={'P10':'P', 'Q10':'Q'})
            .set_index('id')
            .stack(future_stack=True))
        ini_values.index.names = 'id_of_injection', 'part'
        ini_values.name = 'ini'
        # subgraphs
        parts_, factors = get_parts_of_injections(MODEL, step=0, PQlimit=.01)
        parts = parts_.join(ini_values)
        subgraphs_, subgraph_injection_parts, subgraph_batches = \
            get_pq_subgraphs(MODEL, consider_I=False)


        # add initial values for (scaling) factors to subgraph
        subgraph_parts = (
            subgraph_injection_parts
            .join(parts, on=['id_of_injection', 'part']))
        subgraph_part_groupby = (
            subgraph_parts.groupby(['index_of_subgraph', 'part']))
        subgraph_part_val_ini = subgraph_part_groupby[['value', 'ini']].sum()
        not_zero = subgraph_part_val_ini[subgraph_part_val_ini.value != 0]
        k_ini = not_zero.ini / not_zero.value
        k_ini.name = 'k_ini'


        subgraphs = (
            pd.merge(
                left=subgraphs_, right=k_ini.reset_index(['part'], drop=True),
                left_on='index_of_subgraph', right_index=True, how='left')
            .fillna(1.))

        pass

class Make_scaling_factors(unittest.TestCase):

    # def test_empty_model(self):
    #     res = make_scaling_factors(model_from_frames())
    #     pass

    def test_make_scaling_factors(self):
        """Linear, two branches, one injection, one subgraph
        ::
             P=5
             Q=3
            +<-------->+<-------->+----->>
            n0  br00   n1  br01   n2    inj03_0
                                        P10=10
                                        Q10=10

        """
        model = _make_model([
            Slacknode(id_of_node='n0'),
            Branch(
                id='br00', id_of_node_A='n0', id_of_node_B='n1',
                y_lo=1e3+1e3j),
            Branch(
                id='br01', id_of_node_A='n1', id_of_node_B='n2',
                y_lo=1e3+1e3j),
            Injection(id='inj03_0', id_of_node='n2', P10=10, Q10=10),
            #
            PValue(id_of_batch='n0_br00', P=5),
            QValue(id_of_batch='n0_br00', Q=3),
            Output(
                id_of_batch='n0_br00', id_of_node='n0', id_of_device='br00'),
            Factor(id='k'),
            Klink(id_of_factor=('k', 'k'), id_of_injection='inj03_0',
                  part=('p', 'q'))])
        res = make_scaling_factors(model)
        pass


class MultiIndex_slice_test(unittest.TestCase):

    def test_slicing(self):
        """Tests slicing of pandas DataFrame with MultiIndex.

        >>>src
            col
        0 p   a
          q   b
        1 p   c
          q   d
        2 p   e
          q   f

        #slice('0','1'), slice('p')

        >>>expected
            col
        0 p   a
        1 p   c
        """
        src = pd.DataFrame(
            ['a', 'b', 'c', 'd', 'e', 'f'],
            columns=['col'],
            index=pd.MultiIndex.from_product([('0', '1', '2'), ('p', 'q')]))
        expected = pd.DataFrame(
            ['a', 'c'],
            columns=['col'],
            index=pd.MultiIndex.from_product([('0', '1'), ['p']]))
        idx = pd.IndexSlice
        self.assertTrue(
            (src.loc[idx[['0', '1'], ['p']],:] == expected).all().all())

class Scaling_test(unittest.TestCase):

    def test_scaling(self):

        df = pd.DataFrame(
            {'minimal': [25, 0],
             'actual': [50, 60],
             'maximal':[110, 90]})
        df['range'] = df.maximal - df.minimal
        df['actual_normalized'] = (df.actual - df.minimal) / df.range
        k = df.actual_normalized.mean()
        df['offset_k'] = k - df.actual_normalized

        pass



if __name__ == '__main__':
    unittest.main()
