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

Created on Mon Apr 10 14:41:34 2023

@author: pyprg
"""
import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import chain, repeat
from egrid.builder import DEFAULT_FACTOR_ID, Factor, Defk, expand_def
from egrid.topo import get_make_scaling_of_subgraphs

Stepgroups = namedtuple(
    'Stepgroups', 'groups template')
Stepgroups.__doc__ = """pandas.Groupby object and a(n empty) DataFrame
template.

Parameters
----------
groups: pandas.Groupby
    a pandas.DataFrame grouped by columns 'step'

template: pandas.DataFrame
"""

Factors = namedtuple(
    'Factors',
    'gen_factordata gen_injfactor terminalfactors '
    'get_groups get_injfactorgroups')
Factors.__doc__ ="""
Data of generic factors (step == -1),
symbols for generic factors, references to injections and terminals for
generic factors.

Functions to retrieve factors and relations by optimization-steps.

Parameters
----------
* .gen_factordata, pandas.DataFrame (id (str, ID of factor)) ->
    * .step, -1
    * .type, 'var'|'const', type of factor decision variable or parameter
    * .id_of_source, str, id of factor (previous optimization step)
       for initialization
    * .value, float, used by initialization if no source factor in previous
       optimization step
    * .min, float
       smallest possible value
    * .max, float
       greatest possible value
    * .is_discrete, bool
       just 0 digits after decimal point if True, input for solver,
       accepted by MINLP solvers
    * .m, float
       increase of multiplier with respect to change of var/const
       the effective multiplier is a linear function of var/const (mx + n)
    * .n, float
       multiplier when var/const is 0.
       the effective multiplier is a linear function of var/const (mx + n)
    * .index_of_symbol, int
* .gen_injfactor, pandas.DataFrame (id_of_injection, part) ->
    * .step, -1
    * .id, str, ID of factor
* .gen_termfactor, pandas.DataFrame
    * .id_of_branch
    * .id_of_node
    * .id_of_factor
    * .index_of_terminal
    * .index_of_factor
* .terminalfactors, pandas.DataFrame
    * .id, str
    * .index_of_terminal, int
    * .index_of_other_terminal, int
    * .type, 'var'|'const'
    * .id_of_source, str
    * .value, float
    * .min, float
    * .max, float
    * .is_discrete, bool
    * .m, float
    * .n, float
    * .index_of_symbol, int
* .get_groups: function
    (iterable_of_int)-> (pandas.DataFrame)
    ('step', 'id') ->
        * .index, int
        * .type, 'var'|'const', decision variable|parameter
        * .id_of_source, str
        * .value, float
        * .min, float
        * .max, float
        * .is_discrete, bool
        * .m, float
        * .n, float
* .get_injfactorgroups: function
    (iterable_of_int)-> (pandas.DataFrame)
    ('step', 'id_of_injection', 'part') ->
        * .id, str, ID of factor
"""

def _get_factors(injassoc, termassoc, factor_frame, branchterminals):
    """Arranges data of factors for further processing.

    Parameters
    ----------
    injassoc: pandas.DataFrame

    termassoc: pandas.DataFrame

    factor_frame: pandas.DataFrame

    branchterminals: pandas.DataFrame
        * .id_of_node
        * .id_of_branch
        * .index_of_terminal
        * .index_of_other_terminal

    Returns
    -------
    Factors
        * .gen_factordata, pandas.DataFrame
        * .gen_injfactor, pandas.DataFrame
        * .terminalfactors, pandas.DataFrame
        * .get_groups: function
            (iterable_of_int)-> (pandas.DataFrame)
        * .get_injfactorgroups: function
            (iterable_of_int)-> (pandas.DataFrame)"""
    # injection links
    is_valid_injassoc = (
        injassoc
        .reset_index(['step'])
        .set_index(['step', 'id'])
        .join(factor_frame.type, how='left')
        .isin(('const', 'var')))
    is_valid_injassoc.index = injassoc.index
    # terminal links
    is_valid_termassoc = (
        termassoc
        .reset_index(['step'])
        .set_index(['step', 'id'])
        .join(factor_frame.type, how='left')
        .isin(('const', 'var')))
    is_valid_termassoc.index = termassoc.index
    return make_factordefs(
        factor_frame,
        termassoc[is_valid_termassoc.type],
        injassoc[is_valid_injassoc.type],
        branchterminals)

def get_factors(model):
    """Arranges data of factors for further processing.

    Parameters
    ----------
    model: Model

    Returns
    -------
    Factors
        * .gen_factordata, pandas.DataFrame
        * .gen_injfactor, pandas.DataFrame
        * .terminalfactors, pandas.DataFrame
        * .get_groups: function
            (iterable_of_int)-> (pandas.DataFrame)
        * .get_injfactorgroups: function
            (iterable_of_int)-> (pandas.DataFrame)"""
    # factors
    factors_ = model.factors.set_index(['step', 'id'])
    # links of injection
    injassoc_ = model.injectionlinks
    injassoc_ = (
        injassoc_[
            # filter for existing injections
            injassoc_.injid.isin(model.injections.id)
            & injassoc_.part.isin(['p', 'q'])]
        .set_index(['step', 'injid', 'part']))
    injassoc_.index.names = ['step', 'id_of_injection', 'part']
    injassoc = injassoc_[~injassoc_.index.duplicated(keep='first')]
    injindex_ = injassoc.reset_index().groupby(['step', 'id']).any().index
    # links of terminals
    #   filter for existing branchterminals
    termlinks = model.terminallinks
    branchterminals = model.branchterminals
    at_term = (
        pd.MultiIndex.from_frame(termlinks[['branchid', 'nodeid']])
        .isin(
            pd.MultiIndex.from_frame(
                branchterminals[['id_of_branch', 'id_of_node']])))
    termassoc_ = termlinks[at_term].set_index(['step', 'branchid', 'nodeid'])
    termassoc_.index.names=['step', 'id_of_branch', 'id_of_node']
    termassoc = termassoc_[~termassoc_.index.duplicated(keep='first')]
    termindex_ = termassoc.reset_index().groupby(['step', 'id']).any().index
    # filter stepwise for intersection of injlinks+termlinks and factors
    df_ = pd.concat(
        [pd.DataFrame([], index=injindex_),
         pd.DataFrame([], index=termindex_)])
    factor_frame = factors_.join(df_[~df_.index.duplicated()], how='inner')
    return _get_factors(injassoc, termassoc, factor_frame, branchterminals)

def empty_like(df, droplevel=-1):
    """Creates an empty pandas.DataFrame from a template DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame

    droplevel: int
        function drops a level from index if value is greater -1

    Returns
    -------
    pandas.DataFrame"""
    idx_ = df.index.droplevel(droplevel) if -1 < droplevel else df.index
    idx = idx_.delete(range(len(idx_)))
    return pd.DataFrame([], columns=df.columns, index=idx).astype(df.dtypes)

def _create_stepgroups(df):
    """Groups df by column 'step'.

    Parameters
    ----------
    df: pandas.DataFrame
        * .step

    Returns
    -------
    Stepgroups"""
    return Stepgroups(df.groupby('step'), empty_like(df))

def _selectgroup(step, stepgroups):
    """Selects a group with index step from stepgroups. Returns a copy of
    stepgroups.template if no group with index exists.

    Parameters
    ----------
    step: int
        index of optimization step
    stepgroups: Stepgroups

    Returns
    -------
    pandas.DataFrame"""
    try:
        return stepgroups.groups.get_group(step)
    except KeyError:
        return stepgroups.template.copy()

def _selectgroups(stepgroups, steps):
    """Selects and concatenates groups from stepgroups if present.
    Returns an empty pandas.DataFrame otherwise.

    Parameters
    ----------
    stepgroups: Stepgroups

    steps: interable
        int, index of optimization step, first step is 0, generic data
        have step -1

    Returns
    -------
    pandas.DataFrame"""
    return pd.concat([_selectgroup(step, stepgroups) for step in steps])

def make_factordefs(
        factor_frame, terminal_factor_associations,
        injection_factor_associations, branchterminals):
    """Prepares data of factors for processing by steps.

    Parameters
    ----------
    factor_frame: pandas.DataFrame

    terminal_factor_associations: pandas.DataFrame

    injection_factor_associations: pandas.DataFrame

    branchterminals: pandas.DataFrame
        * .id_of_node
        * .id_of_branch
        * .index_of_terminal
        * .index_of_other_terminal'

    Returns
    -------
    Factors
        * .gen_factordata, pandas.DataFrame
        * .gen_injfactor, pandas.DataFrame
        * .terminalfactors, pandas.DataFrame
        * .get_groups: function
            (iterable_of_int)-> (pandas.DataFrame)
        * .get_injfactorgroups: function
            (iterable_of_int)-> (pandas.DataFrame)"""
    factorgroups = _create_stepgroups(factor_frame.reset_index())
    # factors with attribute step == -1
    gen_factors = _selectgroup(-1, factorgroups).reset_index(drop=True)
    # terminal-factor association with step attribute == -1
    termfactorgroups = _create_stepgroups(
        terminal_factor_associations.reset_index())
    gen_termassoc = _selectgroup(-1, termfactorgroups)
    term_to_factor_ = gen_termassoc.drop(columns=['step'])
    term_to_factor = (
        pd.merge(
            left=term_to_factor_,
            right=branchterminals.reset_index()[
                ['id_of_node', 'id_of_branch', 'index_of_terminal',
                 'index_of_other_terminal']],
            left_on=['id_of_node', 'id_of_branch'],
            right_on=['id_of_node', 'id_of_branch'],
            how='inner')
        [['id', 'index_of_terminal', 'index_of_other_terminal']])
    valid_termassoc = gen_termassoc.id.isin(gen_factors.id)
    termassoc_ = gen_termassoc[valid_termassoc]
    # injection-factor association with step attribute == -1
    injfactorgroups = _create_stepgroups(
        injection_factor_associations.reset_index())
    gen_injassoc = _selectgroup(-1, injfactorgroups)
    valid_incassoc = gen_injassoc.id.isin(gen_factors.id)
    injassoc = gen_injassoc[valid_incassoc]
    # only referenced factors are necessary
    valid_factorids = np.sort(pd.concat([termassoc_.id, injassoc.id]).unique())
    factors = (
        gen_factors.set_index('id').reindex(valid_factorids).reset_index())
    factors['index_of_symbol'] = range(len(factors))
    #symbols = _create_symbols_with_ids(factors.id)
    # add index of symbol to termassoc,
    #   terminal factors are NEVER step-specific
    get_groups = lambda steps: (
        _selectgroups(factorgroups, steps)
        .set_index(['step', 'id']))
    get_injfactorgroups = lambda steps: (
        _selectgroups(injfactorgroups, steps)
        .set_index(['step', 'id_of_injection', 'part']))
    gen_factordata = factors.set_index('id')
    terminalfactors = (
        term_to_factor
        .join(gen_factordata[['value', 'm', 'n', 'index_of_symbol']], on='id'))
    return Factors(
        gen_factordata=gen_factordata,
        gen_injfactor=injassoc.set_index(['id_of_injection', 'part']),
        terminalfactors=terminalfactors,
        get_groups=get_groups,
        get_injfactorgroups=get_injfactorgroups)

Factormeta = namedtuple(
    'Factormeta',
    'id_of_step_symbol '
    'index_of_kpq_symbol index_of_var_symbol index_of_const_symbol '
    'values_of_vars values_of_vars_model cost_of_change '
    'var_min var_max is_discrete '
    'values_of_consts '
    'var_const_to_factor var_const_to_kp var_const_to_kq var_const_to_ftaps '
    'id_to_idx')
Factormeta.__doc__="""
Symbols of variables and constants for factors.

Parameters
----------
id_of_step_symbol: str,
    name/id of symbol of one step (without generic symbols)
index_of_kpq_symbol: numpy.array (shape n,2)
    indices of scalingfactors kp, kq for each injection
index_of_var_symbol: numpy.array
    int, indices of decision variables
index_of_const_symbol: numpy.array
    int, indices of parameters
values_of_vars: numpy.array
    column vector, initial values of decision variables for optimizatin step
values_of_vars_model:
    column vector, values of decision variables from model
cost_of_change: numpy.array
    float
var_min: numpy.array
    lower limits of vars
var_max: numpy.array
    upper limits of vars
is_discrete: numpy.array
    bool, flag for variable
values_of_consts: numpy.array
    column vector, values for consts
var_const_to_factor: array_like
    int, index_of_factor=>index_of_var_const
    converts var_const to factor (var_const[var_const_to_factor])
var_const_to_kp: array_like
    int, converts var_const to kp, one active power scaling factor
    for each injection (var_const[var_const_to_kp])
var_const_to_kq: array_like
    int, converts var_const to kq, one reactive power scaling factor
    for each injection (var_const[var_const_to_kq])
var_const_to_ftaps: array_like
    int, converts var_const to ftaps, factor assigned to
    (selected) terminals (var_const[var_const_to_ftaps] selects ftaps
    in the order of model.factors.terminalfactors)
id_to_idx: pandas.Series  (index: id_of_factor)
    int, index_of_symbol"""

def _select_rows(vecs, row_index):
    """Selects rows from vectors.

    Parameters
    ----------
    vecs: iterable
        casadi.SX or casadi.DM, column vector
    row_index: array_like
        int

    Returns
    -------
    iterator
        * casadi.SX / casadi.DM"""
    return (v[row_index, 0] for v in vecs)

def _loc(df, key):
    try:
        return df.loc[key]
    except KeyError:
        return empty_like(df, 0)

def _get_step_injection_part_to_factor(
        injectionids, assoc_frame, indices_of_steps):
    """Arranges ids for all calculation steps and injections.

    Parameters
    ----------
    injectionids: pandas.Series
        str, IDs of all injections
    assoc_frame: (str (step), str (id_of_injection), 'p'|'q' (part))
        * .id, str, ID of factor
    indices_of_steps: array_like
        int, indices of optimization steps

    Returns
    -------
    pandas.Dataframe (int (step), str (id of injection), 'p'|'q' (part))
        * .id, str, identifier of factor"""
    # all injections, create step, id, (pq) for all injections
    index_all = pd.MultiIndex.from_product(
        [indices_of_steps, injectionids, ('p', 'q')],
        names=('step', 'id_of_injection', 'part'))
    # step id_of_injection part => id
    return (
        # do not accept duplicated links
        assoc_frame[~assoc_frame.index.duplicated()]
        .reindex(index_all, fill_value=DEFAULT_FACTOR_ID))

def _get_factor_ini_values(factors):
    """Returns indices for initial values of variables/parameters.

    Parameters
    ----------
    factors: pandas.DataFrame

    symbols: pandas.Series
        int, indices of symbols

    Returns
    -------
    pandas.Series
        int"""
    unique_factors = factors.index
    prev_index = pd.MultiIndex.from_arrays(
        [unique_factors.get_level_values(0) - 1, factors.id_of_source.array])
    ini = factors.index_of_symbol.reindex(prev_index)
    ini.index = unique_factors
    # '-1' means copy initial data from column 'value' as there is no valid
    #   reference to a var/const of previous step
    ini.fillna(-1, inplace=True)
    return ini.astype(dtype='Int64')

def _get_default_factors(indices_of_steps):
    """Generates one default scaling factor for each step.

    The factor is of type 'const' has value 1.0, minimum and maximum
    are 1.0 too.

    Parameters
    ----------
    indices_of_steps: array_like
        int, indices of steps which to create a default factor for

    Parameters
    ----------
    pandas.DataFrame (index: ['step', 'id'])
        value in column 'id' of index is the string
        of egrid.builder.DEFAULT_FACTOR_ID,
        columns according to fields of egrid.builder.Loadfactor"""
    return (
        pd.DataFrame(
            #id type id_of_source value min max is_discrete m n step
            expand_def(
                Defk(
                    id=DEFAULT_FACTOR_ID,
                    type='const',
                    id_of_source=DEFAULT_FACTOR_ID,
                    value=1.0,
                    min=1.0,
                    max=1.0,
                    step=indices_of_steps)),
            columns=Factor._fields)
        .set_index(['step', 'id']))

def _factor_index_per_step(factors, start):
    """Creates an index (0...n) for each step.

    Parameters
    ----------
    factors: pandas.DataFrame (step, id)->...

    start: iterable
        int, first index for each step, if start is None the first index is 0

    Returns
    -------
    pandas.Series"""
    offsets = repeat(0) if start is None else start
    return pd.Series(
        chain.from_iterable(
            (range(off, off+len(factors.loc[step])))
            for off, step in zip(offsets, factors.index.levels[0])),
        index=factors.index,
        name='index_of_symbol',
        dtype=np.int64)

def _get_values_of_symbols(factordata, value_of_previous_step):
    """Returns values for symbols.

    When a symbol is a decision variable the value is used as the initial
    value.

    Values are either given explicitely or are calculated in the previous
    calculation step.

    Parameters
    ----------
    factordata: pandas.DataFrame (sorted according to index_of_symbol)
        * .index_of_symbol, int
        * .value, float
        * .index_of_source, int
    value_of_previous_step: numpy.array
        vector of float

    Returns
    -------
    tuple

        * numpy.array (ordered according to index_of_symbol)
          column vector of float, initial values for step
        * numpy.array (ordered according to index_of_symbol)
          column vector of float, static values of model"""
    # values of model, ordered according to index of symbol
    vals = (
        factordata.value[factordata.index_of_symbol].to_numpy().reshape(-1,1))
    # values for next step
    values = np.zeros((len(factordata),1), dtype=float)
    # fill with explicitely given values not calculated in previous step
    is_given = factordata.index_of_source < 0
    given = factordata[is_given]
    if len(given):
        values[given.index_of_symbol] = given.value.to_numpy().reshape(-1,1)
    # fill with values calculated in previous step
    calc = factordata[~is_given]
    if len(calc):
        assert len(value_of_previous_step), 'missing value_of_previous_step'
        values[calc.index_of_symbol] = (
            value_of_previous_step[calc.index_of_source.astype(int)])
    return values, vals

def _add_step_index(df, step_indices):
    """Copies data of df for each step index in step_indices.

    Concatenates the data.

    Parameters
    ----------
    df: pandas.DataFrame
        df has a MultiIndex
    step_indices: interable
        int

    Returns
    -------
    pandas.DataFrame (extended copy of df)"""
    if df.empty:
        return df.assign(step=0).set_index(['step', df.index])
    # adds a column 'step', adds value 'index_of_step' in each row of
    #   column 'step', then adds the colum 'step' to the index
    #   (column 'step' is added to the old index in the left most position)
    return pd.concat(
        [df.assign(step=idx).set_index(['step', df.index])
         for idx in step_indices])

def _get_injection_factors(step_factor_injection_part, factors):
    """Creates crossreference from injection to scaling factors.

    In particular, creates crossreference from (step, injection) to IDs and
    indices of scaling factors for active and reactive power.

    Parameters
    ----------
    step_factor_injection_part: pandas.DataFrame
        DESCRIPTION.
    factors: pandas.DataFrame
        DESCRIPTION.

    Returns
    -------
    pandas.DataFrame (index: [step, id_of_injection])

        * .id_p
        * .id_q
        * .kp
        * .kq"""
    if not step_factor_injection_part.empty:
        injection_factors = (
            step_factor_injection_part
            .join(factors.index_of_symbol)
            .reset_index()
            .set_index(['step', 'id_of_injection', 'part'])
            .unstack('part')
            .droplevel(0, axis=1))
        injection_factors.columns=['id_p', 'id_q', 'kp', 'kq']
        return injection_factors
    return pd.DataFrame(
        [],
        columns=['id_p', 'id_q', 'kp', 'kq'],
        index=pd.MultiIndex.from_arrays(
            [[],[]], names=['step', 'id_of_injection']))

def _add_default_factors(required_factors):
    if required_factors.isna().any(axis=None):
        # ensure existence of default factors when needed
        default_factor_steps = (
            # type is just an arbitrary column
            required_factors[required_factors.type.isna()]
            .reset_index()
            .step
            .unique())
        default_factors = _get_default_factors(default_factor_steps)
        # replace nan with values (for required default factors)
        return required_factors.combine_first(default_factors)
    return required_factors

def _get_scaling_factor_data(factordefs, injections, steps, start):
    """Creates and arranges data of scaling factors.

    Parameters
    ----------
    factordefs: egrid.factors.Factors
        data of scaling and taps factors
    injections: pandas.DataFrame (index_of_injection)->
        * .id, str, identifier of injection
    steps: iterable
        int, indices of optimization steps, first step has index 0
    start: iterable | None
        int, first index for each step, None: 0 for all steps

    Returns
    -------
    tuple

        * pandas.DataFrame, all scaling factors
          (str (step), 'const'|'var', str (id of factor)) ->

          * .id_of_source, str, source of initial value,
            factor of previous step
          * .value, float, initial value if no valid source reference
          * .min, float, smallest possible value
          * .max, float, greatest possible value
          * .is_discrete, bool, no digits after decimal point if True
            (input for MINLP solver), False for load scaling factors, True
            for capacitor bank model
          * .m, float, not used for scaling factors
          * .n, float, not used for scaling factors
          * .index_of_symbol, int, index in 1d-vector of var/const
          * .index_of_source, int, index in 1d-vector of previous step

        * pandas.DataFrame, injections with scaling factors
          (int (step), str (id_of_injection))

          * .id_p, str, ID for scaling factor of active power
          * .id_q, str, ID for scaling factor of reactive power
          * .kp, int, index of active power scaling factor in 1d-vector
          * .kq, int, index of reactive power scaling factor in 1d-vector
          * .index_of_injection, int, index of affected injection"""
    generic_injfactor_steps = _add_step_index(factordefs.gen_injfactor, steps)
    assoc_steps = factordefs.get_injfactorgroups(steps)
    # get generic_assocs which are not in assocs of step, this allows
    #   overriding the linkage of factors
    assoc_diff = generic_injfactor_steps.index.difference(assoc_steps.index)
    generic_assoc = generic_injfactor_steps.reindex(assoc_diff)
    inj_assoc = pd.concat([generic_assoc, assoc_steps])
    # given factors are either specific for a step or defined for each step
    #   which is indicated by a '-1' step-index
    #   generate factors from those with -1 step setting for given steps
    # step, inj, part => factor
    #   copy generic_factors and add step indices
    generic_factor_steps = _add_step_index(factordefs.gen_factordata, steps)
    # retrieve step specific factors
    factors_steps = factordefs.get_groups(steps)
    # select generic factors only if not part of specific factors
    req_generic_factors_index = generic_factor_steps.index.difference(
        factors_steps.index)
    # union of generic and specific injection factors
    given_factors = pd.concat(
        [generic_factor_steps
             .drop(columns=['index_of_symbol'])
             .reindex(req_generic_factors_index),
         factors_steps])
    # generate MultiIndex for:
    #   - two factors
    #   - all requested steps
    #   - all injections
    step_injection_part__factor = _get_step_injection_part_to_factor(
        injections.id, inj_assoc, steps)
    # step-to-factor index
    step_factor = (
        pd.MultiIndex.from_arrays(
            [step_injection_part__factor.index.get_level_values('step'),
             step_injection_part__factor.id],
            names=['step', 'id'])
        .unique())
    # given_factors, expand union of generic and specific factors to
    #   all required factors
    # this method call destroys type info of column 'is_discrete', was bool
    required_factors = given_factors.reindex(step_factor)
    factors_ = (
        _add_default_factors(required_factors)
        .astype({'is_discrete':bool}, copy=False))
    factors_.sort_index(inplace=True)
    # indices of symbols
    factors_ = factors_.join(
        generic_factor_steps['index_of_symbol'].astype('Int64'),
        how='left')
    no_symbol = factors_['index_of_symbol'].isna()
    if no_symbol.any():
        # range of indices for new scaling factor indices
        factor_index_per_step = _factor_index_per_step(
            factors_[no_symbol], start)
        factors_.loc[no_symbol, 'index_of_symbol'] = factor_index_per_step
    factors = factors_.astype({'index_of_symbol':np.int64})
    # add data for initialization
    factors['index_of_source'] = _get_factor_ini_values(factors)
    injection_factors = _get_injection_factors(
        step_injection_part__factor.reset_index().set_index(['step', 'id']),
        factors)
    # indices of injections ordered according to injection_factors
    injids = injection_factors.index.get_level_values(1)
    index_of_injection = (
        pd.Series(injections.index, index=injections.id)
        .reindex(injids)
        .array)
    injection_factors['index_of_injection'] = index_of_injection
    factors.reset_index(inplace=True)
    factors.set_index(['step', 'type', 'id'], inplace=True)
    return factors, injection_factors

def  _get_taps_factor_data(model_factors, steps):
    """Arranges data of taps factors and values for their initialization.

    Parameters
    ----------
    model_factors: egrid.factors.Factors

    steps: iterable
        int, indices of optimization steps, first step has index 0

    Returns
    -------
    tuple

        * pandas.DataFrame, all taps factors
          (str (step), 'const'|'var', str (id of factor)) ->:

          * .id_of_source, str, source of initial value,
            factor of previous step
          * .value, float, initial value if no valid source reference
          * .min, float, smallest possible value
          * .max, float, greatest possible value
          * .is_discrete, bool, no digits after decimal point if True
            (input for MINLP solver), True for taps model
          * .m, float, -Vdiff per tap-step in case of taps
          * .n, float, n = 1 - (Vdiff * index_of_neutral_position) for taps,
            e.g. when index_of_neutral_position=0 --> n=1
          * .index_of_symbol, int, index in 1d-vector of var/const
          * .index_of_source, int, index in 1d-vector of previous step

        * pandas.DataFrame, terminals with taps factors
          (int (step), str (id_of_factor)):

          * .index_of_terminal, int
          * .index_of_other_terminal, int
          * .value, float, value of var/const
          * .m, float
          * .n, float
          * .index_of_symbol, int"""
    # filter terminal factors from generic factors
    gen_factordata = model_factors.gen_factordata.drop(columns=['step'])
    is_termfactor = gen_factordata.index.isin(model_factors.terminalfactors.id)
    term_factordata = _add_step_index(gen_factordata[is_termfactor], steps)
    # update factors
    #   retrieve step specific factors
    #   step specific factors cannot introduce new taps-factors just
    #   modify generic taps-factors
    factors_of_steps = model_factors.get_groups(steps)
    override_index = term_factordata.index.intersection(factors_of_steps.index)
    cols = [
        'type', 'id_of_source', 'value', 'min', 'max', 'is_discrete', 'm', 'n']
    term_factordata.loc[override_index, cols] = (
        factors_of_steps.loc[override_index, cols])
    # add data for initialization
    term_factordata['index_of_source'] = (
        _get_factor_ini_values(term_factordata))
    terminalfactors = _add_step_index(
        model_factors.terminalfactors.set_index('id'), steps)
    return (
        term_factordata.reset_index().set_index(['step', 'type', 'id']),
        terminalfactors)

def get_factordata_for_step(model, factors, step):
    """Returns data of decision variables and of parameters for a given step.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    factors: Factors
    step: int
        index of optimization step, first index is 0

    Returns
    -------
    tuple

        * count_of_generic_factors, int

        * factors: pandas.DataFrame (for requested step)

            sorted by 'index_of_symbol'
            * .type, 'var|'const', decision variable or
              parameter
            * .id, str, id of factor
            * .id_of_source, id of factor of previous step
              for initialization
            * .value, float, used for initialization if
              id_of_source is invalid, reference for cost of change
            * .min, float, smalles possible value
            * .max, float, greates possible value
            * .is_discrete, bool
            * .m, float, -Vdiff per tap-step in case of taps
            * .n, float, n = 1 - (Vdiff * index_of_neutral_position) for taps,
              e.g. when index_of_neutral_position=0 --> n=1
            * .cost, float, cost of change
            * .index_of_symbol, int, index in 1d-vector of var/const
            * .index_of_source, int, index in 1d-vector of previous step

        * injection_factors: pandas.DataFrame (for each injection),
            ordered by index_of_injection

            * .id_of_injection, str, ID of injection
            * .id_p, str, ID of scaling factor for active power
            * .id_q, str, ID of scaling factor for reactive power
            * .kp, int, index of symbol
            * .kq, int, index of symbol
            * .index_of_injection, int

        * terminal_factors: pandas.DataFrame (for terminals having a factor)

            * .id_of_branch, str, ID of branch
            * .id_of_node, str, ID of node
            * .id, str, ID of factor
            * .index_of_symbol, int
            * .index_of_terminal, int"""
    # index for requested step and the step before requested step,
    #   data of step before are needed for initialization
    steps = [step - 1, step] if 0 < step else [0]
    # factors assigned to terminals
    taps_factors, terminalfactor = _get_taps_factor_data(
        factors, steps)
    # scaling factors for injections
    count_of_generic_factors = len(factors.gen_factordata)
    start = repeat(count_of_generic_factors)
    scaling_factors, injection_factors = _get_scaling_factor_data(
        factors, model.injections, steps, start)
    factors = pd.concat([scaling_factors, taps_factors])
    factors.sort_values('index_of_symbol', inplace=True)
    return (
        count_of_generic_factors,
        _loc(factors, step).reset_index(),
        _loc(injection_factors, step).reset_index().sort_values(
            'index_of_injection'),
        _loc(terminalfactor, step).reset_index())

def _make_factor_meta(
        count_of_generic_factors, factors, injection_factors, terminalfactors,
        k_prev):
    """Prepares data of factors for one step.

    Arguments for solver call:

    Vectors of decision variables ('var') and of paramters ('const'),
    vectors of minimum and of maximum values, vectors of initial values for
    decision variables and values for paramters.

    Data to reorder the solver result:

    Sequences of indices for converting the vector of var/const into the order
    of factors (symbols), into the order of active power scaling factors,
    reactive power scaling factors and taps factors. For getting the
    order of factors (symbols) from var_const (returned by a solver) do:
    ::
        factor[Factormeta.var_const_to_factor] = var_const

    Parameters
    ----------
    count_of_generic_factors: int

        number of generic factors

    factors: pandas.DataFrame

        sorted by 'index_of_symbol'
        * .type, 'var'|'const'
        * .id, str
        * .id_of_source, str
        * .value, float
        * .min, float
        * .max, float
        * .is_discrete, bool
        * .m, float
        * .n, float
        * .index_of_symbol, int
        * .index_of_source, int

    injection_factors: pandas.DataFrame
    (each injection, ordered by index_of_injection)

        * .id_of_injection
        * .id_p
        * .id_q
        * .kp
        * .kq
        * .index_of_injection

    terminalfactors: pandas.DataFrame

        * .id, str
        * .index_of_terminal, int
        * .index_of_other_terminal, int
        * .step, -1
        * .type, 'var'|'const'
        * .id_of_source, str
        * .value, float
        * .min, float
        * .max, float
        * .is_discrete, bool
        * .m, float
        * .n, float
        * .index_of_symbol, int

    k_prev: numpy.array

        float, values of scaling factors from previous step,
        variables and constants

    Returns
    -------
    Factormeta

        id_of_step_symbol: pandas.Series
            str, identifiers for all symbols specific for processed step
        index_of_var_symbol: pandas.Series
            str, inidices of decision variables for processed step
        index_of_const_symbol: pandas.Series
            str, inidices of parameters for processed step
        index_of_kpq_symbol: numpy.aray
            int, shape nx2, scaling factors for active and reactive power
            for all injections
        index_of_ftaps_symbol: pandas.Series
            int, inidices of ftaps-symbols
        index_of_terminal: numpy.array
            int, inidices of terminals having a taps factor
        values_of_vars: numpy.array
            float, column vector, initial values for vars
        values_of_vars_model:
            column vector, values of decision variables from model
        cost_of_change: numpy.array
            float
        var_min: numpy.array
            float, lower limits of vars
        var_max: numpy.array
            float, upper limits of vars
        is_discrete: numpy.array
            bool, flag for variable
        values_of_consts: casadi.DM
            column vector, values for consts
        var_const_to_factor: array_like
            int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
        var_const_to_kp: array_like
            int, converts var_const to kp, one active power scaling factor
            for each injection (var_const[var_const_to_kp])
        var_const_to_kq: array_like
            int, converts var_const to kq, one reactive power scaling factor
            for each injection (var_const[var_const_to_kq])
        var_const_to_ftaps: array_like
            int, converts var_const to ftaps, factor assigned to
            (selected) terminals (var_const[var_const_to_ftaps] selects ftaps
            in the order of argument terminalfactor)
        id_to_idx: pandas.Series  (index: id_of_factor)
            int, index_of_symbol"""
    # inital for vars, value for parameters (consts)
    #   values are ordered by index_of_symbol
    values, values_of_model = _get_values_of_symbols(factors, k_prev)
    factors_var = factors[factors.type=='var']
    values_of_vars = values[factors_var.index_of_symbol,0]
    values_of_vars_model = values_of_model[factors_var.index_of_symbol,0]
    factors_consts = factors[factors.type=='const']
    values_of_consts = values[factors_consts.index_of_symbol,0]
    # the optimization result is provided as a concatenation of
    #   values for decision variables and parameters, here we prepare indices
    #   for mapping to kp/kq (which are ordered according to injections)
    var_const_idxs = (
        np.concatenate(
            [factors_var.index_of_symbol.array,
             factors_consts.index_of_symbol.array])
        .astype(np.int64))
    var_const_to_factor = np.zeros_like(var_const_idxs)
    var_const_to_factor[var_const_idxs] = factors.index_of_symbol
    # step-specific symbols
    id_of_step_symbol = (
        factors.id[count_of_generic_factors <= factors.index_of_symbol])
    id_to_idx = pd.Series(
        factors.index_of_symbol.array,
        index=factors.id,
        name='index_of_symbol')
    return Factormeta(
        id_of_step_symbol=id_of_step_symbol, # per optimization step
        index_of_var_symbol=factors_var.index_of_symbol,
        index_of_const_symbol=factors_consts.index_of_symbol,
        index_of_kpq_symbol=injection_factors[['kp', 'kq']].to_numpy(),
        # initial values, argument in solver call
        values_of_vars=values_of_vars,
        # reference value for cost of change, values of vars from model
        values_of_vars_model=values_of_vars_model,
        cost_of_change=factors_var.cost,
        # lower bound of scaling factors, argument in solver call
        var_min=factors_var['min'],
        # upper bound of scaling factors, argument in solver call
        var_max=factors_var['max'],
        # flag for variable
        is_discrete=factors_var.is_discrete.to_numpy(),
        # values of constants, argument in solver call
        values_of_consts=values_of_consts,
        # reordering of result
        var_const_to_factor=var_const_to_factor,
        var_const_to_kp=var_const_to_factor[injection_factors.kp],
        var_const_to_kq=var_const_to_factor[injection_factors.kq],
        var_const_to_ftaps=var_const_to_factor[
            terminalfactors.index_of_symbol],
        id_to_idx=id_to_idx)

def make_factor_meta(model, factors, step, k_prev):
    """Prepares data of decision variables and paramters for one step.

    Arguments for solver call:

    Vectors of decision variables ('var') and of paramters ('const'),
    vectors of minimum and of maximum values, vectors of initial values for
    decision variables and values for paramters.

    Data for reordering solver result:

    Sequences of indices for converting the vector of var/const into the order
    of factors (symbols), into the order of active power scaling factors,
    reactive power scaling factors and taps factors.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    factors: Factors

    step: int
        index of optimization step, first index is 0
    k_prev: casadi.DM optional
        float, factors of previous optimization step

    Returns
    -------
    Factormeta

        id_of_step_symbol: pandas.Series
            str, identifiers for all symbols specific for processed step
        index_of_var_symbol: pandas.Series
            str, inidices of decision variables for processed step
        index_of_const_symbol: pandas.Series
            str, inidices of parameters for processed step
        index_of_kpq_symbol: numpy.aray
            int, shape nx2, scaling factors for active and reactive power
            for all injections
        index_of_ftaps_symbol: pandas.Series
            int, inidices of ftaps-symbols
        index_of_terminal: numpy.array
            int, inidices of terminals having a taps factor
        values_of_vars: numpy.array
            float, column vector, initial values for vars
        values_of_vars_model:
            column vector, values of decision variables from model
        cost_of_change: numpy.array
            float
        var_min: numpy.array
            float, lower limits of vars
        var_max: numpy.array
            float, upper limits of vars
        is_discrete: numpy.array
            bool, flag for variable
        values_of_consts: casadi.DM
            column vector, values for consts
        var_const_to_factor: array_like
            int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
        var_const_to_kp: array_like
            int, converts var_const to kp, one active power scaling factor
            for each injection (var_const[var_const_to_kp])
        var_const_to_kq: array_like
            int, converts var_const to kq, one reactive power scaling factor
            for each injection (var_const[var_const_to_kq])
        var_const_to_ftaps: array_like
            int, converts var_const to ftaps, factor assigned to
            (selected) terminals (var_const[var_const_to_ftaps])
        id_to_idx: pandas.Series  (index: id_of_factor)
            int, index_of_symbol"""
    return _make_factor_meta(
        *get_factordata_for_step(model, factors, step), k_prev)

# factor generation

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
    factors: pandas.DataFrame
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
    links.sort_values(['step'])
    return (
        links.drop('step', axis=1)
        .loc[~links.loc[:,['injid', 'part', 'id']].duplicated(keep='last')])

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
    injs.rename(columns={'P10': 'p', 'Q10':'q'}, inplace=True)
    parts_ = injs.stack(future_stack=True)
    parts_.rename('value', inplace=True)
    is_significant = PQlimit < parts_
    is_significant.rename('is_significant', inplace=True)
    parts = pd.concat([parts_, is_significant], axis=1)
    parts.index.rename(['id_of_injection', 'part'], inplace=True)
    return parts

def get_parts_of_injections(model, *, step, PQlimit):
    """Retrieves all parts of injections.

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
        * pandas.DataFrame (index: index_of_injection, part)
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
    injection_parts = (
        pd.merge(
            left=significant_parts, right=var_type, how='left',
            left_index=True, right_index=True)
        .fillna('var'))
    injection_parts['is_scalable'] = (
        injection_parts.is_significant & (injection_parts.var_type == 'var'))
    injection_parts['positive_value'] =  0 <= injection_parts.value
    return injection_parts, factors_of_step

def get_pq_subgraphs(model, consider_I=False):
    """Splits graph at P/Q/I values. Collects data of subgraphs.

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
        * .scaling_type, 'P'|'Q'
        * .id_of_injection, str
    subgraph_batches: pandas.DataFrame
        * .id_of_batch, str
        * .P, bool
        * .Q, bool
        * .I, bool
        * .index_of_subgraph, int
        * .scaling_type, 'P'|'Q'"""
    index_of_subgraph = 0
    injection_dfs = []
    batches_dfs = []
    has_slack_ = []
    make_scaling_of_subgraphs = get_make_scaling_of_subgraphs(model)
    for scaling_type in 'PQ':
        barrier_types = (
            [scaling_type] + ['I'] if consider_I else [scaling_type])
        for injections_batches in (
                make_scaling_of_subgraphs(barrier_types)):
            injections, batches, has_slack = injections_batches
            injection_dfs.append(pd.DataFrame(
                {'index_of_subgraph': index_of_subgraph,
                 'scaling_type': scaling_type,
                 'id_of_injection': injections.index}))
            batches['index_of_subgraph'] = index_of_subgraph
            batches['scaling_type'] = scaling_type
            batches_dfs.append(batches)
            index_of_subgraph += 1
            has_slack_.append((index_of_subgraph, has_slack))      
    graph_injections = pd.concat(injection_dfs).reset_index(drop=True)       
    graph_batches = pd.concat(batches_dfs).reset_index(drop=True)
    graph_df = pd.DataFrame(
        has_slack_, columns =['index_of_subgraph', 'has_slack'])
    return graph_df, graph_injections, graph_batches

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
    injection_parts, factors = get_parts_of_injections(
        model, step=step, PQlimit=PQlimit)
    scalable_parts = injection_parts[injection_parts.is_scalable]
    
    
    subgraphs, subgraph_injections, subgraph_batches = get_pq_subgraphs(
        model, consider_I)
    



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



