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

Created on Fri Aug 25 21:03:29 2023

@author: pyprg
"""

import context
import unittest
import networkx as nx
import egrid.builder as grid
from numpy.testing import assert_array_equal
from networkx.algorithms import bipartite
from egrid import _make_model
from egrid.topo import (
    get_node_device_graph, split, get_make_subgraphs, 
    get_make_scaling_of_subgraphs, get_outputs, get_batches_with_type)

def _get_linear_model():
    nodes_ids = list(str(i) for i in range(10))
    pairs = zip(nodes_ids[:-1], nodes_ids[1:])
    branches = [
        grid.Branch(id=f'{a}-{b}', id_of_node_A=a, id_of_node_B=b)
        for a, b in pairs]
    return _make_model(branches)

class Get_node_branch_graph(unittest.TestCase):

    _linear_model = _get_linear_model()

    def test_empty(self):
        digraph = get_node_device_graph(_make_model(()))
        self.assertIsInstance(digraph, nx.DiGraph)
        self.assertTrue(bipartite.is_bipartite(digraph))
        self.assertTrue(nx.is_empty(digraph))

    def test_is_bipartite(self):
        digraph = get_node_device_graph(self._linear_model)
        cns, branches = bipartite.sets(digraph)
        cn_ids = (
            set(self._linear_model.branchterminals.id_of_node)
            | set(self._linear_model.bridgeterminals.id_of_node))
        self.assertEqual(cns, cn_ids)
        self.assertTrue(bipartite.is_bipartite(digraph))

    # def test_split(self):
    #     digraph = get_node_device_graph(self._linear_model)
    #     bridgeterminals = self._linear_model.bridgeterminals.iloc[[6]]
    #     terminal = bridgeterminals.iloc[0]
    #     id_of_node = terminal.id_of_node
    #     id_of_branch = terminal.id_of_branch
    #     edge = (id_of_node, id_of_branch)
    #     self.assertIn(edge, digraph.edges)
    #     subgraphs = list(split(digraph, terminals=[edge]))
    #     self.assertEqual(len(subgraphs), 2)
    #     connectivity_nodes = set()
    #     branches = set()
    #     for sub in subgraphs:
    #         self.assertTrue(bipartite.is_bipartite(sub))
    #         cn, br = bipartite.sets(sub)
    #         connectivity_nodes |= cn
    #         branches |= br
    #         self.assertGreater(len(cn), 0)
    #         self.assertGreater(len(br), 0)
    #         self.assertNotIn((id_of_node, id_of_branch), sub.edges)
    #     na, nb = bipartite.sets(digraph)
    #     self.assertEqual(connectivity_nodes, na)
    #     self.assertEqual(branches, nb)

subgraph_model = _make_model([
    grid.Slacknode(id_of_node='n0'),
    grid.Branch(
        id='br00', id_of_node_A='n0', id_of_node_B='n1', y_lo=1e3+1e3j),
    grid.Branch(
        id='br01', id_of_node_A='n1', id_of_node_B='n2', y_lo=1e3+1e3j),
    grid.Branch(
        id='br02', id_of_node_A='n2', id_of_node_B='n3', y_lo=1e3+1e3j),
    grid.Branch(
        id='br02_2', id_of_node_A='n2', id_of_node_B='n3_2', y_lo=1e3+1e3j),
    grid.Injection(id='inj02_0', id_of_node='n2'),
    grid.Injection(id='inj02_1', id_of_node='n2', P10=1),
    grid.Injection(id='inj02_2', id_of_node='n2', P10=1, Q10=1),
    grid.Injection(id='inj02_3', id_of_node='n2', P10=1, Q10=1),
    grid.Injection(id='inj02_4', id_of_node='n2', P10=1, Q10=1),
    grid.Injection(id='inj02_5', id_of_node='n2', P10=1, Q10=1),
    grid.Branch(
        id='br03', id_of_node_A='n3', id_of_node_B='n4'),
    grid.Branch(
        id='br04', id_of_node_A='n4', id_of_node_B='n5', y_lo=1e3+1e3j),
    grid.PValue(id_of_batch='n2_br02', P=5),
    grid.QValue(id_of_batch='n2_br02', Q=3),
    grid.Output(
        id_of_batch='n2_br02', id_of_node='n2', id_of_device='br02'),
    grid.Output(
        id_of_batch='n2_br02', id_of_node='n2', id_of_device='br02_2'),
    grid.Output(id_of_batch='n2_br02', id_of_device='inj02'),
    grid.Factor(id='kp'),
    grid.Factor(id='kp_const', type='const'),
    grid.Factor(id='kp0', step=0),
    grid.Factor(id='kq'),
    grid.Klink(id_of_factor='kp_const', id_of_injection='inj02_5', part='p'),
    grid.Klink(id_of_factor='kp', id_of_injection='inj02_2', part='p'),
    grid.Klink(id_of_factor='kp', id_of_injection='inj02_3', part='p', step=0),
    grid.Klink(id_of_factor='kq', id_of_injection='inj02_3', part='q', step=0),
    grid.Klink(id_of_factor='kp', id_of_injection='inj02_4', part='p'),
    grid.Klink(id_of_factor='kp0', id_of_injection='inj02_4', part='p', step=0),
    grid.Klink(id_of_factor='kq0', id_of_injection='inj02_4', part='q', step=0)])

class Get_make_subgraphs(unittest.TestCase):

    def test_make_subgraphs_PQ(self):
        outputs = get_outputs(subgraph_model)
        batches_with_type =  get_batches_with_type(subgraph_model, outputs)
        make_subgraphs = get_make_subgraphs(
            subgraph_model, outputs, batches_with_type)
        subgraphs = list(make_subgraphs(['P', 'Q']))
        self.assertEqual(len(subgraphs), 2)
        subgraphs.sort(key=lambda x:len(x[1]))
        edges0, cns0, devs0 = subgraphs[0]
        self.assertEqual(cns0, {'n0', 'n1', 'n2'})
        self.assertEqual(
            devs0,
            {'inj02_5', 'inj02_4', 'inj02_0', 'br00', 'inj02_1', 'br01',
              'inj02_2', 'inj02_3'})
        edges1, cns1, devs1 = subgraphs[1]
        self.assertEqual(cns1, {'n5', 'n3', 'n4', 'n3_2', 'n2_br02'})
        self.assertEqual(devs1, {'br04', 'br02_2', 'br02', 'br03'})

class Get_make_scaling_of_subgraphs(unittest.TestCase):

    def test_get_make_scaling_of_subgraphs(self):
        make_scaling_of_subgraphs = get_make_scaling_of_subgraphs(
            subgraph_model)
        scaling_of_subgraphs = list(
            make_scaling_of_subgraphs(flow_types=['P', 'Q']))
        scaling0, scaling1 = scaling_of_subgraphs
        self.assertEqual(len(scaling0[1]), 1)
        self.assertEqual(len(scaling1[1]), 1)

# class Get_injection_groups(unittest.TestCase):

#     def test_get_injection_groups(self):
#         make_subgraphs = get_make_subgraphs(_subgraph_model)
#         group_info, injection_info = get_injection_groups(
#             make_subgraphs(['P', 'Q']))
#         group_info.sort_values('index_of_group', inplace=True)
#         assert_array_equal(
#             group_info.to_numpy(), [[0, False, True], [1, True, False]])
#         self.assertEqual(injection_info.id_of_injection[0], 'inj02')
#         self.assertEqual(injection_info.index_of_group[0], 1)

if __name__ == '__main__':
    unittest.main()
