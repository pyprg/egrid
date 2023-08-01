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
import pandas as pd
from egrid.model import model_from_frames
from egrid.builder import make_data_frames, create_objects
from egrid.check import check_frames, get_first_error

def make_model(*args):
    """Creates an instance of egrid.Model.

    Parameters
    ----------
    args: iterable
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
        Vlimit, Defk, Deft, Defoterm, Klink, Tlink, str
        strings in args are processed with graphparser.parse

    Returns
    -------
    egrid.Model"""
    frames = make_data_frames(create_objects(args))
    return model_from_frames(frames)

def make_model_checked(*args):
    """Creates an instance of egrid.Model. Checks data.

    Parameters
    ----------
    args: iterable
        Branch, Slacknode, Injection, Output, PValue, QValue, IValue, Vvalue,
        Vlimit, Defk, Deft, Defoterm, Klink, Tlink, str
        strings in args are processed with graphparser.parse

    Returns
    -------
    egrid.Model"""
    frames = make_data_frames(create_objects(args))
    frames['Message'] = pd.concat([
        frames['Message'],
        pd.DataFrame.from_records(
            check_frames(frames), columns=['message','level'])])
    return model_from_frames(frames)
