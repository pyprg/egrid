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

Created on Sun Dec 10 16:38:56 2023

@author: pyprg


This module is about automatic generation of scaling factors for injections.
The graph of the grid is split at flow measurements P/Q/I into subgraphs.
Functions of this module will create scaling factors for active and reactive
power per each subgraph if apropriate injection for scaling are part of a
particular subgraph."""
import pandas as pd
import numpy as np
from egrid.topo import get_make_subgraphs_with_batches
from egrid._types import DEFAULT_FACTOR_ID

def get_factors_of_step(factors, step):
    """Collects generic and step-specific data of factors.

    When functions of this model are used for generation of scaling factors,
    injections are scaled by default in the range of 0...infinity. Other
    options are:
        * no scaling at all (constant)
        * different range
    None-default scaling factors can be defined and linked to parts (P/Q)
    of injections explicitely. This function fetches explicitely defined
    factors from the model. Returned factors are defined for each step
    (value of step is -1) or for the step addressed by argument 'step'.
    Generic factors are overruled by factors of 'step'.

    Parameters
    ----------
    factors: pandas.DataFrame
        * .id, str
        * .step, int
    step: int
        index of optimization step

    Returns
    -------
    pandas.DataFrame"""
    myfactors = factors.loc[factors.step.isin([-1, step])]
    myfactors.sort_values(['step'], inplace=True)
    return (
        myfactors.drop('step', axis=1)
        .loc[~myfactors.id.duplicated(keep='last')])

def get_injectionlinks_of_step(injectionlinks, step):
    """Collects generic and step-specific injectionlinks.

    Parameters
    ----------
    injectionlinks: pandas.DataFrame
        * .injid, str
        * .part, 'p'|'q'
        * .id, str
        * .step, int
    step: int
        index of optimization step

    Returns
    -------
    pandas.DataFrame"""
    links = injectionlinks.loc[injectionlinks.step.isin([-1, step])]
    links['part'] = links.part.apply(str.upper)
    links = links.sort_values(['injid', 'part', 'step']).drop('step', axis=1)
    return links.loc[~links.loc[:,['injid', 'part']].duplicated(keep='last')]

def get_significant_parts(injections, PQlimit):
    """The function returns all parts accompanied by property 'is_significant'.

    Parts are active power and reactive power of injections. A part is
    significant if its value is greater than PQlimit.

    Evaluates P10 and Q10.

    Parameterters
    -------------
    injections: pandas.DataFrame (index: id_of_injection, part)
        * .P10, float, active power at V_abs = 1.0
        * .Q10, float, reactive power at V_abs = 1.0

    Returns
    -------
    pandas.DataFrame ('id_of_injection', 'part')
        * .is_significant
        * .value"""
    if injections.empty:
        return pd.DataFrame(
            [],
            columns=['value', 'is_significant'],
            index=pd.MultiIndex.from_arrays(
                [[], []],
                names=['id_of_injection', 'part']))
    injs = injections.set_index('id').loc[:,['P10', 'Q10']]
    injs.rename(columns={'P10': 'P', 'Q10':'Q'}, inplace=True)
    parts_ = injs.stack(future_stack=True)
    parts_.rename('value', inplace=True)
    is_significant = PQlimit < parts_
    is_significant.rename('is_significant', inplace=True)
    parts = pd.concat([parts_, is_significant], axis=1)
    parts.index.rename(['id_of_injection', 'part'], inplace=True)
    return parts

def get_parts_of_injections(model, *, step, PQlimit):
    """Retrieves all parts of injections (active power P and reactive power Q).

    The function provides data on active and reactive power (parts)
    for each injection of the given model.

    Default values used if no values given by a factor definitions for step:
        * .var_type = 'var'
        * .min = 0
        * .max = inf
        * .is_discrete = False

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    step: int
        index of optimization step
    PQlimit: float
        minimum value of P and Q for scalable parts

    Returns
    -------
    tuple
        * pandas.DataFrame (index: id_of_injection, part)
            * .is_significant, bool
            * .var_type, 'var'|'const'
            * .is_scalable, bool
            * .value
        * pandas.DataFrame
            * .id, str, identifier of injection
            * .type, 'var'|'const'
            * .id_of_source, identifier of factor for initialization
            * .value, float
            * .min, float
            * .max, float
            * .is_discrete, bool
            * .m, float
            * .n, float
            * .cost, float"""
    injectionlinks_of_step = get_injectionlinks_of_step(
        model.injectionlinks, step)
    factors_of_step_ = get_factors_of_step(model.factors, step)
    factors_of_step = factors_of_step_.loc[
        factors_of_step_.id.isin(injectionlinks_of_step.id)]
    var_type = (
        pd.merge(
            left=injectionlinks_of_step,
            right=factors_of_step[['id', 'type', 'min', 'max', 'is_discrete']],
            left_on='id', right_on='id')
        .set_index(['injid', 'part']))
    var_type.rename(
        columns={'type': 'var_type', 'id': 'id_of_factor'}, inplace=True)
    var_type.index.rename(['id_of_injection', 'part'], inplace=True)
    significant_parts = get_significant_parts(model.injections, PQlimit)
    injection_parts = pd.merge(
            left=significant_parts, right=var_type, how='left',
            left_index=True, right_index=True)
    injection_parts.id_of_factor.fillna(DEFAULT_FACTOR_ID, inplace=True)
    injection_parts.var_type.fillna('var', inplace=True)
    injection_parts['min'].fillna(0, inplace=True)
    injection_parts['max'].fillna(np.inf, inplace=True)
    injection_parts.is_discrete.fillna(False, inplace=True)
    injection_parts['is_scalable'] = (
        injection_parts.is_significant & (injection_parts.var_type == 'var'))
    injection_parts['positive_value'] = 0 <= injection_parts.value
    if (any(injection_parts.id_of_factor == DEFAULT_FACTOR_ID) and
           not any(factors_of_step.id == DEFAULT_FACTOR_ID)):
        factors_of_step = pd.concat(
            [factors_of_step,
             # default scaling, if not given explicitely
             pd.DataFrame(
                 dict(
                     id=[DEFAULT_FACTOR_ID],
                     type=['var'],
                     id_of_source=[DEFAULT_FACTOR_ID],
                     value=[1.], min=[0], max=[np.inf],
                     is_discrete=[False], m=[1.], n=[0.], cost=[1.]))],
            ignore_index=True)
    return injection_parts, factors_of_step

def _get_pq_subgraphs(model, *, consider_I=False):
    """Splits graph at P/Q/I values. Collects data of subgraphs.

    Separate subgraphs are created for independent scaling of P and Q.
    P-values are barriers for P-subgraph splitting, Q-values for Q-subgraphs.
    I-values are considered additional barriers for splitting of both types of
    subgraphs if consider_I is True .

    Parameters
    ----------
    model: egrid.model.Model
        data of an electric grid
    consider_I: bool, optional
        terminals with electric current values are subgraph borders,
        the default is false

    Returns
    -------
    subgraphs: pandas.DataFrame
        * .index_of_subgraph, int
        * .has_slack, bool
    subgraph_injections: pandas.DataFrame
        * .index_of_subgraph, int
        * .part, 'P'|'Q'
        * .id_of_injection, str
    subgraph_batches: pandas.DataFrame
        * .id_of_batch, str
        * .P, bool, has PValue
        * .Q, bool, has QValue
        * .I, bool, has IValue
        * .index_of_subgraph, int
        * .part, 'P'|'Q'"""
    index_of_subgraph = 0
    injection_dfs = []
    batches_dfs = []
    graph = []
    make_subgraphs_with_batches = get_make_subgraphs_with_batches(model)
    for scaling_type in 'PQ':
        barrier_types = (
            [scaling_type] + ['I'] if consider_I else [scaling_type])
        for injections, batches, has_slack in (
                make_subgraphs_with_batches(barrier_types)):
            injection_dfs.append(pd.DataFrame(
                {'index_of_subgraph': index_of_subgraph,
                 'part': scaling_type,
                 'id_of_injection': injections.index}))
            # consider batches with suitable barrier type only
            if not batches.empty and \
                any(batches[bt][0] for bt in barrier_types):
                batches['index_of_subgraph'] = index_of_subgraph
                batches['part'] = scaling_type
                batches_dfs.append(batches)
            graph.append((index_of_subgraph, has_slack, scaling_type))
            index_of_subgraph += 1
    graph_injections = (
        pd.concat(injection_dfs).reset_index(drop=True)
        if injection_dfs else
        pd.DataFrame(
            [],
            columns=['index_of_subgraph', 'part', 'id_of_injection'])
        .astype(
            {'index_of_subgraph':np.int64,
             'part':str,
             'id_of_injection':str}))
    graph_batches = (
        pd.concat(batches_dfs).reset_index(drop=True)
        if batches_dfs else
        pd.DataFrame(
            [],
            columns=[
                'id_of_batch', 'P', 'Q', 'I', 'index_of_subgraph',
                'scaling_type'])
        .astype(
            {'id_of_batch':str, 'P':bool, 'Q':bool, 'I':bool,
             'index_of_subgraph':np.int64, 'scaling_type':str}))
    graph_df = (
        pd.DataFrame(
            graph,
            columns=['index_of_subgraph', 'has_slack', 'scaling_type'])
        .astype({'index_of_subgraph':np.int64, 'has_slack':bool}))
    return graph_df, graph_injections, graph_batches

def get_pq_subgraphs(
        model, *, ini_values=None, consider_I=False, PQlimit=.01):
    """Creates P/Q-subgraphs for unique scaling factors.

    The method is suitable for splitting a network graph in parts comprising
    one active power and one reactive power scaling factor for all connected
    injections. The function can be used for the first optimization step
    as it accepts initial values for injections separately.

    Batches for I/P/Q including the same branch terminals / injection terminals
    shall have the same ID.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    ini_value: pandas.Series (index: ['id_of_injection', 'part']), optional
        initial values of parts (P or Q). The default is None.
    consider_I: bool, optional
        terminals with electric current values are subgraph borders,
        default is False
    PQlimit: float, optional
        minimum value of P and Q for scalable parts, default is .01.

    Returns
    -------
    subgraphs: pandas.DataFrame
        * .index_of_subgraph, int
        * .has_slack, bool
        * .scaling_type, 'P'|'Q'
        * .k_ini, float, initial value of scaling factor
    subgraph_parts: pandas.DataFrame
        * .index_of_subgraph, int
        * .part, 'P'|'Q'
        * .id_of_injection, str, unique idendifier of injection
        * .value, float, P10|Q10
        * .is_significant, bool, true size exceeds PQlimit
        * .id_of_factor, str, scaling factor
        * .var_type, 'var'|'const'
        * .min, float, smallest possible value
        * .max, float, greatest possible value
        * .is_discrete, bool, int value if true else float
        * .is_scalable, bool, var_type=='var' and is_significant
        * .positive_value, bool, 0 <= value
        * .ini, float, initial value
    subgraph_batches: pandas.DataFrame
        * .id_of_batch, str
        * .P, bool, has PValue
        * .Q, bool, has QValue
        * .I, bool, has IValue
        * .index_of_subgraph, int
        * .part, 'P'|'Q'"""
    if ini_values is None:
        # initial values (test)
        ini_values = (
            model.injections
            .loc[:,['id','P10','Q10']]
            .rename(columns={'P10':'P', 'Q10':'Q'})
            .set_index('id')
            .stack(future_stack=True))
        ini_values.index.names = 'id_of_injection', 'part'
        ini_values.name = 'ini'
    # subgraphs
    subgraphs_, subgraph_injection_parts, subgraph_batches = \
        _get_pq_subgraphs(model, consider_I=consider_I)
    # enhance subgraph_injection_parts (factors is not used)
    parts_, factors = get_parts_of_injections(model, step=0, PQlimit=.01)
    parts = parts_.join(ini_values)
    subgraph_parts = (
        subgraph_injection_parts
        .join(parts, on=['id_of_injection', 'part']))
    # enhance subgraph with initial scaling factor
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
    return subgraphs, subgraph_parts, subgraph_batches

def add_subgraph_properties(subgraphs, subgraph_parts, subgraph_batches):
    """Enhances subgraphs with aggregated properties.

    The properties control the generation of scaling factors.

    'add_subgraph_properties' implements a heuristic approach.
    'add_subgraph_properties' creates scaling factors for each distinct
    combination of the load properties 'P/Q-part', 'index_of_subgraph',
    'positive_value' and 'is_discrete'.

    Minimum and Maximum of scaling factors consider the sums of minimum and
    maximum of all load parts of the concerning group. Thus, it is possible
    that the product of kmin and scheduled active power is smaller then
    individual minimum multiplied by active power value for one specific load.
    This applies accordingly to kmax.

    Parameters
    ----------
    subgraphs: pandas.DataFrame
        * .index_of_subgraph
        * .has_slack
        * .scaling_type
        * .k_ini
    subgraph_parts: pandas.DataFrame
        * .index_of_subgraph
        * .part
        * .id_of_injection
        * .value
        * .is_significant
        * .id_of_factor
        * .var_type
        * .min
        * .max
        * .is_discrete
        * .is_scalable
        * .positive_value
        * .ini.
    subgraph_batches: pandas.DataFrame
        * .id_of_batch
        * .P
        * .Q
        * .I
        * .index_of_subgraph
        * .part
        * .has_part

    Returns
    -------
    pandas.DataFrame
        ['index_of_subgraph', 'positive_value ', 'is_discrete'] is unique

        * .index_of_subgraph, int
        * .positive_value, bool, injections have positive P/Q-value
        * .min_value
        * .max_value
        * .sum_of_values
        * .k_min
        * .k_max
        * .is_scalable, bool, scaling of P/Q is possible
        * .fixed, bool, P/Q is given at all borders (not just I)
        * .has_slack, bool, subgraph contains a slack node
        * .scaling_type, 'P'|'Q', for scaling of active or reactive power
        * .k_ini, float, initial scaling factor"""
    scalable_parts = subgraph_parts[subgraph_parts.is_scalable]
    # range of injection values calculated from range of scaling factors
    # min(value*min, value*max), max(value*min, value*max)
    scalable_parts[['min_value', 'max_value']] = np.apply_along_axis(
        lambda x: (np.min(x), np.max(x)),
        axis=1,
        arr=scalable_parts['value'].to_numpy()
            # minimum / maximum of scaling factors
            * scalable_parts[['min', 'max']].to_numpy())
    subgraph_parts_props = (
        scalable_parts
        .groupby(['index_of_subgraph', 'positive_value', 'is_discrete'])
        .agg(
            min_value=('min_value', 'sum'),
            max_value=('max_value', 'sum'),
            sum_of_values=('value', 'sum')))
    subgraph_parts_props[['k_min', 'k_max']] = (
        subgraph_parts_props[['min_value', 'max_value']].to_numpy()
        / subgraph_parts_props.sum_of_values.to_numpy())
    subgraph_parts_props['is_scalable'] = True
    subgraph_parts_props.reset_index(inplace=True)
    # properties of measured flow values / flow setpoint
    #   select P or Q depending of 'part'
    subgraph_batches['has_part'] = subgraph_batches.apply(
        lambda row:row[row.part], axis=1)
    subgraph_properties = (
        subgraph_batches
        .groupby('index_of_subgraph')
        # fixed might be false if I is involved
        #   or there is no value for scaling_type at all,
        #   if scaling_type is not fixed the OBJECTIVE function needs
        #   a term to determin the scaling factor in case the
        #   subgraph has scalable parts of scaling_type
        .agg(fixed=('has_part', all)))
    properties = pd.merge(
        left=subgraph_parts_props,
        right=subgraph_properties,
        left_on='index_of_subgraph',
        right_index=True)
    fill = dict(
        min_value=0, max_value=np.inf, sum_of_values=0, k_min=0, k_max=np.inf,
        is_discrete=False, fixed=False, positive_value=False,
        negative_value=False, is_scalable=False)
    sg = (
        pd.merge(
            left=properties,
            right=subgraphs,
            left_on='index_of_subgraph',
            right_on='index_of_subgraph',
            how='outer')
        .fillna(fill))
    sg.loc[sg.has_slack, ['is_scalable']] = False
    return sg

def make_scaling_factors(model, *, step=0, PQlimit=.01, consider_I=False):
    """Produces scaling factors for one step.

    Splits graph at flow measurements (or setpoints, which is at locations of
    known P/Q/I-values). Makes scaling factors for subgraphs. Creates
    separate subgraphs for P/Q. Creates scaling factors if there are
    appropriate parts for P/Q-scaling in the subgraph only. Injections
    are scaled by default in the range of 0...inifinity. Scaling can be
    overruled by explicitely defined and linked factors for the purpose
    of a modified scaling range or for avoiding scaling at all if  parts of
    injections are constant.

    Parameters
    ----------
    model: egrid.model.Model
        data of an electric grid
    step: int, optional
        index of optimization step, the default is 0
    PQlimit: float, optional
        minimum power for scaling, the default is .01
    consider_I: bool, optional
        splits P/Q-graphs additionally at location of I values,
        the default is false

    Returns
    -------
    tuple
        * pandas.DataFrame, scaling_factors (index_of_scalingfactor)
            * .is_discrete
            * .k_min
            * .k_max
            * .k_ini
            * .name_of_scaling_factor
        * scalable_parts
            * .id_of_injection
            * .part
            * .index_of_scalingfactor"""
    subgraphs, subgraph_parts, subgraph_batches = get_pq_subgraphs(
        model, consider_I=consider_I, PQlimit=PQlimit)
    sg = add_subgraph_properties(subgraphs, subgraph_parts, subgraph_batches)
    sg['name_of_scaling_factor'] = '-'
    scalable_sg = sg[sg.is_scalable]
    scalable_sg.loc[:, ['name_of_scaling_factor']] = (
        'k' + scalable_sg.scaling_type.str.lower()
        + sg.index[sg.is_scalable].to_series().apply(str)
        + scalable_sg.positive_value.apply(lambda b: 'p' if b else 'n')
        + scalable_sg.is_discrete.apply(lambda b: 'd' if b else 'c'))
    scalable_sg.reset_index(inplace=True, names='index_of_scalingfactor')
    sg_parts = pd.merge(
        right=subgraph_parts,
        left=scalable_sg[
            ['index_of_subgraph', 'scaling_type', 'name_of_scaling_factor',
             'index_of_scalingfactor']],
        right_on=['index_of_subgraph', 'part'],
        left_on=['index_of_subgraph', 'scaling_type'],
        how='inner')
    return (
        scalable_sg.loc[
            :,
            ['is_discrete',  'k_min', 'k_max', 'k_ini',
             'name_of_scaling_factor', 'index_of_scalingfactor']]
        .set_index('index_of_scalingfactor'),
        sg_parts.loc[:, ['id_of_injection', 'part', 'index_of_scalingfactor']])

