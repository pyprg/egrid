# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:07:21 2022

@author: pyprg
"""
import unittest
import src.egrid.gridmodel as gridmodel

class Gridmodel(unittest.TestCase):
    
    def test_make_model_empty(self):
        gridmodel.make_model()

if __name__ == '__main__':
    unittest.main()