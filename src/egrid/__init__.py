# -*- coding: utf-8 -*-
"""
Interface of egrid
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

Created on Mon Jan 31 08:00:31 2022

@author: pyprg
"""
from .model import model_from_frames
from .builder import make_data_frames, create_objects

def make_model(*args):
    """Creates an instance of gridmodel.Model.

    Parameters
    ----------
    args: iterable
        Branch, Slacknode, Injection, Output, PQValue, IValue, Vvalue,
        Branchtaps, Defk, Link, str
        strings in args are processed with graphparser.parse

    Returns
    -------
    model.Model"""
    frames = make_data_frames(create_objects(args))
    return model_from_frames(frames)
