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
"""
import pandas as pd
import numpy as np
from egrid.topo import get_make_scaling_of_subgraphs
from egrid._types import DEFAULT_FACTOR_ID

def get_factors_of_step(factors, step):
    """Collects generic and step-specific data of factors.

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
    """

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
    injection_parts['min'].fillna(-np.inf, inplace=True)
    injection_parts['max'].fillna(np.inf, inplace=True)
    injection_parts.is_discrete.fillna(False, inplace=True)
    injection_parts['is_scalable'] = (
        injection_parts.is_significant & (injection_parts.var_type == 'var'))
    injection_parts['positive_value'] =  0 <= injection_parts.value
    if (any(injection_parts.id_of_factor == DEFAULT_FACTOR_ID) and
           not any(factors_of_step.id == DEFAULT_FACTOR_ID)):
        factors_of_step = pd.concat(
            [factors_of_step,
             pd.DataFrame(
                 dict(
                     id=[DEFAULT_FACTOR_ID],
                     type=['var'],
                     id_of_source=[DEFAULT_FACTOR_ID],
                     value=[1.], min=[-np.inf], max=[np.inf],
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
    make_scaling_of_subgraphs = get_make_scaling_of_subgraphs(model)
    for scaling_type in 'PQ':
        barrier_types = (
            [scaling_type] + ['I'] if consider_I else [scaling_type])
        for injections_batches in (
                make_scaling_of_subgraphs(barrier_types)):
            injections, batches, has_slack = injections_batches
            injection_dfs.append(pd.DataFrame(
                {'index_of_subgraph': index_of_subgraph,
                 'part': scaling_type,
                 'id_of_injection': injections.index}))
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
        _get_pq_subgraphs(model, consider_I=False)
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

def make_scaling_factors(model, *, step=0, PQlimit=.01, consider_I=False):
    """Produces scaling factors.

    Parameters
    ----------
    model: egrid.model.Model
        data of an electric grid
    step: int, optional
        index of optimization step, the default is 0
    PQlimit: float, optional
        minimum power for scaling, the default is .01
    consider_I: bool, optional
        the default is false

    Returns
    -------
    pandas.DataFrame"""

    subgraphs, subgraph_injection_parts, subgraph_batches = _get_pq_subgraphs(
        model, consider_I=consider_I)

    injection_parts, factors = get_parts_of_injections(
        model, step=step, PQlimit=PQlimit)

    subgraph_parts = subgraph_injection_parts.join(
        injection_parts[injection_parts.is_scalable],
        on=['id_of_injection', 'part'])
    subgraph_parts['min_value'] = subgraph_parts['value'] * subgraph_parts['min']
    subgraph_parts['max_value'] = subgraph_parts['value'] * subgraph_parts['max']
    sg_props = (
        subgraph_parts
        .groupby('index_of_subgraph')
        .agg(
            min_value=('min_value', 'min'),
            max_value=('max_value', 'max'),
            positive_value=('positive_value', any),
            negative_value=('positive_value', lambda series: any(~series)))
        )

    # properties of measured flow values / flow setpoint
    #   select P or Q depending of 'part'
    subgraph_batches['has_part'] = subgraph_batches.apply(
        lambda row:row[row.part],axis=1)
    subgraph_properties = (
        subgraph_batches
        .groupby('index_of_subgraph')
        .agg(fixed=('has_part', all), part=('part', 'first')))

            # # scaling_type_fixed might be false if I is involved
            # #   or there is no value for scaling_type at all,
            # #   if scaling_type is not fixed the objective function needs
            # #   a term to determin the scaling factor in case the
            # #   subgraph has scalable parts of scaling_type
            # scaling_type_fixed = all(subgraph_batches[scaling_type])
            # # scalable_subgraph_parts of correct scaling_type
            # scalable_subgraph_parts = scalable_parts.loc[
            #     pd.IndexSlice[
            #         subgraph_injections.index, scaling_type.lower()],:]
            # # has_scalable_part is true if the subgraph contains an injection
            # #   having a scalable part of scaling_type (P/Q)
            # has_scalable_part = not scalable_subgraph_parts.empty
            # subgraph_batches.sort_values('id_of_batch', inplace=True)
            # # table 0:
            # #   scaling_type, index_of_subgraph, fixed
            # # table 1:
            # #   index_of_subgraph, part_of_injection
    pass



