# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:24:50 2023

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
import context
from egrid._types import (Defoterm, expand_defoterm, Term)

class Expand_defoterm(unittest.TestCase):

    def test_default(self):
        defoterm = Defoterm()
        res = list(expand_defoterm(0, defoterm))
        expected = [Term(id='0')]
        self.assertEqual(res, expected)

    def test_fn(self):
        defoterm = Defoterm(fn='avg')
        res = list(expand_defoterm(0, defoterm))
        expected = [Term(id='0', fn='avg')]
        self.assertEqual(res, expected)

    def test_fn_weight(self):
        defoterm = Defoterm(fn='avg', weight=3.14)
        res = list(expand_defoterm(0, defoterm))
        expected = [Term(id='0', fn='avg', weight=3.14)]
        self.assertEqual(res, expected)

    def test_single_argument(self):
        defoterm = Defoterm(args='a')
        res = list(expand_defoterm(0, defoterm))
        expected = [Term(id='0', args=['a'])]
        self.assertEqual(res, expected)

    def test_two_argument(self):
        defoterm = Defoterm(args=('a', 'b'))
        res = list(expand_defoterm(0, defoterm))
        expected = [Term(id='0', args=('a', 'b'))]
        self.assertEqual(res, expected)

    def test_with_step(self):
        defoterm = Defoterm(args='a', step=0)
        res = list(expand_defoterm(0, defoterm))
        expected = [Term(id='0', args=['a'], step=0)]
        self.assertEqual(res, expected)

    def test_with_two_steps(self):
        defoterm = Defoterm(args='a', step=(0, 2))
        res = list(expand_defoterm(0, defoterm))
        expected = [
            Term(id='0', args=['a'], step=0),
            Term(id='0', args=['a'], step=2)]
        self.assertEqual(res, expected)

if __name__ == '__main__':
    unittest.main()
