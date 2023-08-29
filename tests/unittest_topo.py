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
from networkx.algorithms import bipartite
from egrid import make_model
from egrid.topo import (
    get_node_device_graph, split, get_make_subgraphs, get_injection_groups)

def _get_linear_model():
    nodes_ids = list(str(i) for i in range(10))
    pairs = zip(nodes_ids[:-1], nodes_ids[1:])
    branches = [
        grid.Branch(id=f'{a}-{b}', id_of_node_A=a, id_of_node_B=b)
        for a, b in pairs]
    return make_model(branches)

class Get_node_branch_graph(unittest.TestCase):

    _linear_model = _get_linear_model()

    def test_empty(self):
        digraph = get_node_device_graph(make_model())
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

    def test_split(self):
        digraph = get_node_device_graph(self._linear_model)
        bridgeterminals = self._linear_model.bridgeterminals.iloc[[6]]
        terminal = bridgeterminals.iloc[0]
        id_of_node = terminal.id_of_node
        id_of_branch = terminal.id_of_branch
        edge = (id_of_node, id_of_branch)
        self.assertIn(edge, digraph.edges)
        subgraphs = list(split(digraph, terminals=[edge]))
        self.assertEqual(len(subgraphs), 2)
        connectivity_nodes = set()
        branches = set()
        for sub in subgraphs:
            self.assertTrue(bipartite.is_bipartite(sub))
            cn, br = bipartite.sets(sub)
            connectivity_nodes |= cn
            branches |= br
            self.assertGreater(len(cn), 0)
            self.assertGreater(len(br), 0)
            self.assertNotIn((id_of_node, id_of_branch), sub.edges)
        na, nb = bipartite.sets(digraph)
        self.assertEqual(connectivity_nodes, na)
        self.assertEqual(branches, nb)

class Get_make_subgraphs(unittest.TestCase):

    def test_make_subgraphs(self):
        elements =[
            grid.Slacknode(id_of_node='n0'),
            grid.Branch(
                id='br00', id_of_node_A='n0', id_of_node_B='n1', y_lo=1e3+1e3j),
            grid.Branch(
                id='br01', id_of_node_A='n1', id_of_node_B='n2', y_lo=1e3+1e3j),
            grid.Branch(
                id='br02', id_of_node_A='n2', id_of_node_B='n3', y_lo=1e3+1e3j),
            grid.Branch(
                id='br02_2', id_of_node_A='n2', id_of_node_B='n3_2', y_lo=1e3+1e3j),
            grid.Injection(id='inj02', id_of_node='n2', P10=1),
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
            grid.Output(id_of_batch='n2_br02', id_of_device='inj02')]
        model = make_model(elements)
        make_subgraphs = get_make_subgraphs(model)

        group_info, injection_info = get_injection_groups(
            make_subgraphs(['P', 'Q']))
        pass

if __name__ == '__main__':
    unittest.main()
