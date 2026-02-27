"""
PF AI Lab 5.0 — Optuna NSGA-II Optimizer
Focused mode (~12 params) and Full mode (~25 params).

Performance:
  - Optuna trials: n_jobs=1 (sequential — fastest due to Python GIL)
  - Post-processing: ProcessPoolExecutor for true multi-core parallelism
    (ThreadPoolExecutor is SLOWER than sequential for CPU-bound Python code)
"""

import optuna
import numpy as np
import pandas as pd
import time
import os
import math
import warnings
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import (
    FOCUSED_SEARCH_SPACE, FOCUSED_FIXED_PARAMS,
    FULL_SEARCH_SPACE, OPTIMIZER_DEFAULTS,
    ROBUSTNESS_BONUS, MIN_TRADES, get_default_config,
    PERTURBATION_PCT, PERTURBATION_N_RUNS,
)
from .backtest_engine import run_backtest
from .metrics import compute_metrics, compute_composite_score
from .ml.walk_forward import run_walk_forward

# Keys used to build a trial fingerprint for exclusion matching
_FINGERPRINT_KEYS = (
    'entry_type', 'pmax_length', 'pmax_multiplier', 'pmax_ma_type',
    'pmax_ma_mode', 'pmax_ma_factor', 'pmax_ma_distance',
    'use_ma_cross_signal', 'sig_ma_type', 'sig_ma_fast', 'sig_ma_slow',
    'don_length', 'rsi_div_length', 'rsi_div_pivot', 'rsi_div_hidden',
    'fractal_buffer', 'length_bb_ma', 'mult_bb_ma', 'length_bb_sma',
    'use_lt_filter', 'lt_multiplier',
    'use_mt_filter',
    'use_rsi50_filter',
    'use_diff_ma_red', 'diff_ma_red_pct',
    # V5.2 risk management
    'sl_mode', 'sl_pct', 'atr_sl_mult', 'sl_dynamic',
    'tp_mode', 'tp_pct', 'tp_sl_ratio',
    'be_mode', 'be_trigger_pct', 'tsl_pct',
    'max_hold_days',
    'use_adx_regime_filter', 'adx_trend_threshold',
    'use_vol_filter', 'vol_percentile_threshold',
    'long_side', 'short_side',
    # V5.3 Tier 3
    'use_partial_tp', 'tp1_pct', 'tp1_size', 'tp2_pct', 'tp2_size',
    'use_session_filter', 'session1_start', 'session1_end',
    'use_vix_filter', 'vix_threshold_high',
    'use_macro_filter', 'macro_min', 'macro_max',
    'use_qq_filter', 'qq_length',
)

# Suppress Optuna's verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _safe_best_value(study):
    """Get study.best_value without crashing when no trials are completed."""
    try:
        return study.best_value
    except ValueError:
        return -999.0


# ==============================================================================
# TOP-LEVEL FUNCTIONS for ProcessPoolExecutor (must be pickle-able)
# ==============================================================================

def _run_oos_backtest_worker(args):
    """Worker: run OOS backtest for a single config. Top-level for pickle."""
    config, df_oos, instrument = args
    try:
        trades_oos = run_backtest(df_oos, config, instrument)
        return compute_metrics(trades_oos)
    except Exception:
        return {'n_trades': 0, 'calmar': 0, 'profit_factor': 0, 'max_dd': 0}


def _perturbation_worker(args):
    """Worker: run perturbation robustness check. Top-level for pickle."""
    config, df_is, instrument = args
    return _check_perturbation_robustness(config, df_is, instrument)


def _walk_forward_worker(args):
    """Worker: run walk-forward analysis. Top-level for pickle."""
    config, df, instrument = args
    return run_walk_forward(
        df=df, config=config, instrument=instrument,
        n_windows=5, is_pct=0.60, overlap=0.5
    )


def _parallel_optuna_worker(args):
    """
    Worker: run a batch of Optuna trials in a SEPARATE PROCESS.
    Each process has its own GIL → true CPU parallelism.
    Returns a list of {config, score, metrics} dicts.

    NOTE: No shared state (multiprocessing.Value etc.) — Windows uses
    'spawn' which cannot inherit synchronized objects via submit().
    Progress is tracked per-worker-completion in the parent process.

    args[7] (forced_entry_type): if not None, restricts this worker to a
    single entry type (used for Quota Mode).
    args[8] (entry_choices): list of entry types for the search space.
    args[9] (locked_params): dict of {param: value} to lock.
    """
    df_is, instrument, mode, n_trials, seed, excluded_fps, excluded_configs, forced_entry_type, entry_choices, locked_params = args

    # Patch search space entry_type choices for this process
    if entry_choices:
        FOCUSED_SEARCH_SPACE['entry_type']['choices'] = entry_choices
        FULL_SEARCH_SPACE['entry_type']['choices'] = entry_choices

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, df_is, instrument, mode,
                                excluded_fps=excluded_fps,
                                excluded_configs=excluded_configs,
                                forced_entry_type=forced_entry_type,
                                locked_params=locked_params),
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1,
    )

    results = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            results.append({
                'config': t.user_attrs.get('config', {}),
                'score': t.value,
                'metrics': t.user_attrs.get('metrics', {}),
            })
    return results


# ==============================================================================
# PARAMETER SUGGESTION
# ==============================================================================

def suggest_params(trial: optuna.Trial, search_space: dict,
                   fixed_params: dict, mode: str = 'focused') -> dict:
    """
    Suggest parameters from Optuna trial based on the search space.
    Returns a complete config dict.
    """
    config = get_default_config()

    # Apply fixed params first
    config.update(fixed_params)

    # Entry type
    entry_type = trial.suggest_categorical('entry_type', search_space['entry_type']['choices'])
    config['entry_type'] = entry_type

    # PMax
    config['pmax_length'] = trial.suggest_int('pmax_length',
                                               search_space['pmax_length']['low'],
                                               search_space['pmax_length']['high'])
    config['pmax_multiplier'] = trial.suggest_float('pmax_multiplier',
                                                     search_space['pmax_multiplier']['low'],
                                                     search_space['pmax_multiplier']['high'],
                                                     step=search_space['pmax_multiplier'].get('step', 0.1))

    if mode == 'full' and 'pmax_ma_type' in search_space:
        config['pmax_ma_type'] = trial.suggest_categorical('pmax_ma_type',
                                                            search_space['pmax_ma_type']['choices'])

    # PMax MA mode (4 modes)
    if 'pmax_ma_mode' in search_space:
        ma_mode = trial.suggest_categorical('pmax_ma_mode',
                                             search_space['pmax_ma_mode']['choices'])
        config['pmax_ma_mode'] = ma_mode
        if ma_mode in ('mult_int', 'mult_dec') and 'pmax_ma_factor' in search_space:
            config['pmax_ma_factor'] = trial.suggest_float('pmax_ma_factor',
                                                            search_space['pmax_ma_factor']['low'],
                                                            search_space['pmax_ma_factor']['high'],
                                                            step=search_space['pmax_ma_factor'].get('step', 0.01))
        elif ma_mode == 'additive' and 'pmax_ma_distance' in search_space:
            config['pmax_ma_distance'] = trial.suggest_float('pmax_ma_distance',
                                                              search_space['pmax_ma_distance']['low'],
                                                              search_space['pmax_ma_distance']['high'],
                                                              step=search_space['pmax_ma_distance'].get('step', 0.5))

    # Independent MA Cross Signal
    if 'use_ma_cross_signal' in search_space:
        use_ma_cross = trial.suggest_categorical('use_ma_cross_signal',
                                                   search_space['use_ma_cross_signal']['choices'])
        config['use_ma_cross_signal'] = use_ma_cross
        if use_ma_cross:
            if 'sig_ma_type' in search_space:
                config['sig_ma_type'] = trial.suggest_categorical('sig_ma_type',
                                                                    search_space['sig_ma_type']['choices'])
            if 'sig_ma_fast' in search_space:
                config['sig_ma_fast'] = trial.suggest_int('sig_ma_fast',
                                                            search_space['sig_ma_fast']['low'],
                                                            search_space['sig_ma_fast']['high'])
            if 'sig_ma_slow' in search_space:
                config['sig_ma_slow'] = trial.suggest_int('sig_ma_slow',
                                                            search_space['sig_ma_slow']['low'],
                                                            search_space['sig_ma_slow']['high'])

    # Entry params
    if entry_type == 'Donc+Chikou':
        config['don_length'] = trial.suggest_int('don_length',
                                                  search_space['don_length']['low'],
                                                  search_space['don_length']['high'])
    elif entry_type == 'RSI Divergence':
        config['rsi_div_length'] = trial.suggest_int('rsi_div_length',
                                                      search_space['rsi_div_length']['low'],
                                                      search_space['rsi_div_length']['high'])
        config['rsi_div_pivot'] = trial.suggest_int('rsi_div_pivot',
                                                     search_space['rsi_div_pivot']['low'],
                                                     search_space['rsi_div_pivot']['high'])
        config['rsi_div_hidden'] = trial.suggest_categorical('rsi_div_hidden',
                                                              search_space['rsi_div_hidden']['choices'])
    elif entry_type == 'Fractal':
        if 'fractal_buffer' in search_space:
            config['fractal_buffer'] = trial.suggest_float('fractal_buffer',
                                                            search_space['fractal_buffer']['low'],
                                                            search_space['fractal_buffer']['high'],
                                                            step=search_space['fractal_buffer'].get('step', 0.5))
    elif entry_type == 'Boll+SMA':
        if 'length_bb_ma' in search_space:
            config['length_bb_ma'] = trial.suggest_int('length_bb_ma',
                                                        search_space['length_bb_ma']['low'],
                                                        search_space['length_bb_ma']['high'])
        if 'mult_bb_ma' in search_space:
            config['mult_bb_ma'] = trial.suggest_float('mult_bb_ma',
                                                        search_space['mult_bb_ma']['low'],
                                                        search_space['mult_bb_ma']['high'],
                                                        step=search_space['mult_bb_ma'].get('step', 0.1))
        if 'length_bb_sma' in search_space:
            config['length_bb_sma'] = trial.suggest_int('length_bb_sma',
                                                         search_space['length_bb_sma']['low'],
                                                         search_space['length_bb_sma']['high'])
    elif entry_type == 'MM Cross':
        if 'mm_cross_ma1_type' in search_space:
            config['mm_cross_ma1_type'] = trial.suggest_categorical('mm_cross_ma1_type',
                                                                      search_space['mm_cross_ma1_type']['choices'])
        if 'mm_cross_ma1_len' in search_space:
            config['mm_cross_ma1_len'] = trial.suggest_int('mm_cross_ma1_len',
                                                             search_space['mm_cross_ma1_len']['low'],
                                                             search_space['mm_cross_ma1_len']['high'])
        if 'mm_cross_ma2_type' in search_space:
            config['mm_cross_ma2_type'] = trial.suggest_categorical('mm_cross_ma2_type',
                                                                      search_space['mm_cross_ma2_type']['choices'])
        if 'mm_cross_ma2_len' in search_space:
            config['mm_cross_ma2_len'] = trial.suggest_int('mm_cross_ma2_len',
                                                             search_space['mm_cross_ma2_len']['low'],
                                                             search_space['mm_cross_ma2_len']['high'])

    # Filters
    use_lt = trial.suggest_categorical('use_lt_filter', search_space['use_lt_filter']['choices'])
    config['use_lt_filter'] = use_lt
    if use_lt and 'lt_multiplier' in search_space:
        config['lt_multiplier'] = trial.suggest_categorical('lt_multiplier',
                                                             search_space['lt_multiplier']['choices'])

    use_mt = trial.suggest_categorical('use_mt_filter', search_space['use_mt_filter']['choices'])
    config['use_mt_filter'] = use_mt

    config['use_rsi50_filter'] = trial.suggest_categorical('use_rsi50_filter',
                                                            search_space['use_rsi50_filter']['choices'])

    use_diff = trial.suggest_categorical('use_diff_ma_red', search_space['use_diff_ma_red']['choices'])
    config['use_diff_ma_red'] = use_diff
    if use_diff and 'diff_ma_red_pct' in search_space:
        config['diff_ma_red_pct'] = trial.suggest_float('diff_ma_red_pct',
                                                         search_space['diff_ma_red_pct']['low'],
                                                         search_space['diff_ma_red_pct']['high'],
                                                         step=search_space['diff_ma_red_pct'].get('step', 0.5))

    # Full mode additional params
    if mode == 'full':
        # --- LT Trend (unlocked in Full) ---
        if use_lt and 'lt_tf' in search_space:
            config['lt_tf'] = trial.suggest_categorical('lt_tf',
                                                         search_space['lt_tf']['choices'])
        if use_lt and 'lt_length' in search_space:
            config['lt_length'] = trial.suggest_int('lt_length',
                                                     search_space['lt_length']['low'],
                                                     search_space['lt_length']['high'])
        if use_lt and 'lt_ma_type' in search_space:
            config['lt_ma_type'] = trial.suggest_categorical('lt_ma_type',
                                                              search_space['lt_ma_type']['choices'])
        # --- MT Trend (unlocked in Full) ---
        if use_mt and 'mt_tf' in search_space:
            config['mt_tf'] = trial.suggest_categorical('mt_tf',
                                                         search_space['mt_tf']['choices'])
        if use_mt and 'mt_length' in search_space:
            config['mt_length'] = trial.suggest_int('mt_length',
                                                     search_space['mt_length']['low'],
                                                     search_space['mt_length']['high'])
        if use_mt and 'mt_ma_type' in search_space:
            config['mt_ma_type'] = trial.suggest_categorical('mt_ma_type',
                                                              search_space['mt_ma_type']['choices'])
        if use_mt and 'mt_multiplier' in search_space:
            config['mt_multiplier'] = trial.suggest_categorical('mt_multiplier',
                                                                 search_space['mt_multiplier']['choices'])

        if 'use_ma_dir_filter' in search_space:
            use_ma_dir = trial.suggest_categorical('use_ma_dir_filter',
                                                    search_space['use_ma_dir_filter']['choices'])
            config['use_ma_dir_filter'] = use_ma_dir
            if use_ma_dir and 'ma_dir_len' in search_space:
                config['ma_dir_len'] = trial.suggest_int('ma_dir_len',
                                                          search_space['ma_dir_len']['low'],
                                                          search_space['ma_dir_len']['high'])

        if 'use_mavg_filter' in search_space:
            config['use_mavg_filter'] = trial.suggest_categorical('use_mavg_filter',
                                                                   search_space['use_mavg_filter']['choices'])

        if 'order_pause' in search_space:
            config['order_pause'] = trial.suggest_int('order_pause',
                                                       search_space['order_pause']['low'],
                                                       search_space['order_pause']['high'])

    # ==================================================================
    # V5.2: Risk Management parameters (both Focused and Full modes)
    # ==================================================================

    # SL Mode
    if 'sl_mode' in search_space:
        sl_mode = trial.suggest_categorical('sl_mode',
                                              search_space['sl_mode']['choices'])
        config['sl_mode'] = sl_mode

        if sl_mode == 'static_pct' and 'sl_pct' in search_space:
            config['sl_pct'] = trial.suggest_float('sl_pct',
                                                     search_space['sl_pct']['low'],
                                                     search_space['sl_pct']['high'],
                                                     step=search_space['sl_pct'].get('step', 1.0))

        if sl_mode == 'atr_based' and 'atr_sl_mult' in search_space:
            config['atr_sl_mult'] = trial.suggest_float('atr_sl_mult',
                                                          search_space['atr_sl_mult']['low'],
                                                          search_space['atr_sl_mult']['high'],
                                                          step=search_space['atr_sl_mult'].get('step', 0.5))
            if 'sl_dynamic' in search_space:
                sl_dyn = trial.suggest_categorical('sl_dynamic',
                                                     search_space['sl_dynamic']['choices'])
                config['sl_dynamic'] = sl_dyn
                if sl_dyn:
                    if 'sl_vol_min_mult' in search_space:
                        config['sl_vol_min_mult'] = trial.suggest_float('sl_vol_min_mult',
                                                                          search_space['sl_vol_min_mult']['low'],
                                                                          search_space['sl_vol_min_mult']['high'],
                                                                          step=search_space['sl_vol_min_mult'].get('step', 0.1))
                    if 'sl_vol_max_mult' in search_space:
                        config['sl_vol_max_mult'] = trial.suggest_float('sl_vol_max_mult',
                                                                          search_space['sl_vol_max_mult']['low'],
                                                                          search_space['sl_vol_max_mult']['high'],
                                                                          step=search_space['sl_vol_max_mult'].get('step', 0.1))

    # TP Mode
    if 'tp_mode' in search_space:
        tp_mode = trial.suggest_categorical('tp_mode',
                                              search_space['tp_mode']['choices'])
        config['tp_mode'] = tp_mode
        if tp_mode == 'static_pct' and 'tp_pct' in search_space:
            config['tp_pct'] = trial.suggest_float('tp_pct',
                                                     search_space['tp_pct']['low'],
                                                     search_space['tp_pct']['high'],
                                                     step=search_space['tp_pct'].get('step', 1.0))
        if tp_mode == 'sl_ratio' and 'tp_sl_ratio' in search_space:
            config['tp_sl_ratio'] = trial.suggest_float('tp_sl_ratio',
                                                          search_space['tp_sl_ratio']['low'],
                                                          search_space['tp_sl_ratio']['high'],
                                                          step=search_space['tp_sl_ratio'].get('step', 0.5))

    # BE/TSL Mode
    if 'be_mode' in search_space:
        be_mode = trial.suggest_categorical('be_mode',
                                              search_space['be_mode']['choices'])
        config['be_mode'] = be_mode
        if be_mode != 'no' and 'be_trigger_pct' in search_space:
            config['be_trigger_pct'] = trial.suggest_float('be_trigger_pct',
                                                             search_space['be_trigger_pct']['low'],
                                                             search_space['be_trigger_pct']['high'],
                                                             step=search_space['be_trigger_pct'].get('step', 0.5))
        if be_mode == 'be_tsl' and 'tsl_pct' in search_space:
            config['tsl_pct'] = trial.suggest_float('tsl_pct',
                                                      search_space['tsl_pct']['low'],
                                                      search_space['tsl_pct']['high'],
                                                      step=search_space['tsl_pct'].get('step', 0.5))

    # Max Hold Days
    if 'max_hold_days' in search_space:
        config['max_hold_days'] = trial.suggest_int('max_hold_days',
                                                      search_space['max_hold_days']['low'],
                                                      search_space['max_hold_days']['high'])

    # ADX Regime Filter
    if 'use_adx_regime_filter' in search_space:
        use_adx = trial.suggest_categorical('use_adx_regime_filter',
                                              search_space['use_adx_regime_filter']['choices'])
        config['use_adx_regime_filter'] = use_adx
        if use_adx and 'adx_trend_threshold' in search_space:
            config['adx_trend_threshold'] = trial.suggest_int('adx_trend_threshold',
                                                                search_space['adx_trend_threshold']['low'],
                                                                search_space['adx_trend_threshold']['high'])

    # Volatility ATR Percentile Filter
    if 'use_vol_filter' in search_space:
        use_vol = trial.suggest_categorical('use_vol_filter',
                                              search_space['use_vol_filter']['choices'])
        config['use_vol_filter'] = use_vol
        if use_vol and 'vol_percentile_threshold' in search_space:
            config['vol_percentile_threshold'] = trial.suggest_int('vol_percentile_threshold',
                                                                      search_space['vol_percentile_threshold']['low'],
                                                                      search_space['vol_percentile_threshold']['high'])

    # Long/Short Side
    if 'long_side' in search_space:
        config['long_side'] = trial.suggest_categorical('long_side',
                                                          search_space['long_side']['choices'])
    if 'short_side' in search_space:
        config['short_side'] = trial.suggest_categorical('short_side',
                                                           search_space['short_side']['choices'])
    # Guard: at least one side must be active
    if not config.get('long_side', True) and not config.get('short_side', True):
        config['long_side'] = True

    # ==================================================================
    # V5.3: Tier 3 parameters (both Focused and Full modes)
    # ==================================================================

    # Partial Profit Taking
    if 'use_partial_tp' in search_space:
        use_ptp = trial.suggest_categorical('use_partial_tp',
                                              search_space['use_partial_tp']['choices'])
        config['use_partial_tp'] = use_ptp
        if use_ptp:
            if 'tp1_pct' in search_space:
                config['tp1_pct'] = trial.suggest_float('tp1_pct',
                                                          search_space['tp1_pct']['low'],
                                                          search_space['tp1_pct']['high'],
                                                          step=search_space['tp1_pct'].get('step', 0.5))
            if 'tp1_size' in search_space:
                config['tp1_size'] = trial.suggest_int('tp1_size',
                                                         search_space['tp1_size']['low'],
                                                         search_space['tp1_size']['high'],
                                                         step=search_space['tp1_size'].get('step', 10))
            if 'tp2_pct' in search_space:
                config['tp2_pct'] = trial.suggest_float('tp2_pct',
                                                          search_space['tp2_pct']['low'],
                                                          search_space['tp2_pct']['high'],
                                                          step=search_space['tp2_pct'].get('step', 0.5))
            if 'tp2_size' in search_space:
                config['tp2_size'] = trial.suggest_int('tp2_size',
                                                         search_space['tp2_size']['low'],
                                                         search_space['tp2_size']['high'],
                                                         step=search_space['tp2_size'].get('step', 10))

    # Session / Timing Filter
    if 'use_session_filter' in search_space:
        use_sess = trial.suggest_categorical('use_session_filter',
                                               search_space['use_session_filter']['choices'])
        config['use_session_filter'] = use_sess
        if use_sess:
            if 'session1_start' in search_space:
                config['session1_start'] = trial.suggest_int('session1_start',
                                                                search_space['session1_start']['low'],
                                                                search_space['session1_start']['high'],
                                                                step=search_space['session1_start'].get('step', 100))
            if 'session1_end' in search_space:
                config['session1_end'] = trial.suggest_int('session1_end',
                                                              search_space['session1_end']['low'],
                                                              search_space['session1_end']['high'],
                                                              step=search_space['session1_end'].get('step', 100))

    # VIX Filter (Security Layer 0)
    if 'use_vix_filter' in search_space:
        use_vix = trial.suggest_categorical('use_vix_filter',
                                              search_space['use_vix_filter']['choices'])
        config['use_vix_filter'] = use_vix
        if use_vix and 'vix_threshold_high' in search_space:
            config['vix_threshold_high'] = trial.suggest_int('vix_threshold_high',
                                                                search_space['vix_threshold_high']['low'],
                                                                search_space['vix_threshold_high']['high'],
                                                                step=search_space['vix_threshold_high'].get('step', 5))

    # Macro Filter Simplifie
    if 'use_macro_filter' in search_space:
        use_macro = trial.suggest_categorical('use_macro_filter',
                                                search_space['use_macro_filter']['choices'])
        config['use_macro_filter'] = use_macro
        if use_macro:
            if 'macro_min' in search_space:
                config['macro_min'] = trial.suggest_float('macro_min',
                                                            search_space['macro_min']['low'],
                                                            search_space['macro_min']['high'],
                                                            step=search_space['macro_min'].get('step', 10.0))
            if 'macro_max' in search_space:
                config['macro_max'] = trial.suggest_float('macro_max',
                                                            search_space['macro_max']['low'],
                                                            search_space['macro_max']['high'],
                                                            step=search_space['macro_max'].get('step', 10.0))

    # QQ Estimation Filter
    if 'use_qq_filter' in search_space:
        use_qq = trial.suggest_categorical('use_qq_filter',
                                             search_space['use_qq_filter']['choices'])
        config['use_qq_filter'] = use_qq
        if use_qq and 'qq_length' in search_space:
            config['qq_length'] = trial.suggest_int('qq_length',
                                                       search_space['qq_length']['low'],
                                                       search_space['qq_length']['high'],
                                                       step=search_space['qq_length'].get('step', 7))

    return config


# ==============================================================================
# OBJECTIVE FUNCTION
# ==============================================================================

def _config_fingerprint(config: dict) -> str:
    """
    Build a deterministic fingerprint string from the strategy-relevant
    parameters of a config.  Used to detect duplicates.
    """
    parts = []
    for k in sorted(_FINGERPRINT_KEYS):
        if k in config:
            v = config[k]
            if isinstance(v, float):
                v = round(v, 1)
            parts.append(f'{k}={v}')
    return '|'.join(parts)


def _is_too_similar(config_fp: str, excluded_fps: set,
                    config: dict, excluded_configs: list) -> bool:
    """
    Check if a trial config matches (exact) or is too close to any
    previously explored config.

    1. Exact fingerprint match → exclude
    2. "Neighbourhood" match: same entry_type + PMax within ±1 length
       and ±0.2 multiplier → exclude (avoids trivial re-exploration)
    """
    # --- Exact match ---
    if config_fp in excluded_fps:
        return True

    # --- Neighbourhood match ---
    cfg_entry = config.get('entry_type', '')
    cfg_plen = config.get('pmax_length', 0)
    cfg_pmul = config.get('pmax_multiplier', 0.0)

    for exc in excluded_configs:
        if exc.get('entry_type') != cfg_entry:
            continue
        exc_plen = exc.get('pmax_length', -999)
        exc_pmul = exc.get('pmax_multiplier', -999.0)
        if (abs(cfg_plen - exc_plen) <= 1 and
                abs(cfg_pmul - exc_pmul) <= 0.2):
            # Same entry + nearly identical PMax → too similar
            return True

    return False


def objective(trial: optuna.Trial, df_is: pd.DataFrame,
              instrument: str, mode: str = 'focused',
              excluded_fps: set = None,
              excluded_configs: list = None,
              forced_entry_type: str = None,
              locked_params: dict = None) -> float:
    """
    Optuna objective function.
    Returns composite score (to maximize).

    If `excluded_fps` / `excluded_configs` are provided, trials that
    match previously-explored parameter combinations receive a heavy
    penalty (-998) so Optuna learns to avoid that region.

    If `forced_entry_type` is provided (Quota Mode), the entry_type
    is fixed instead of being sampled by Optuna.

    If `locked_params` is provided, those parameters are overridden
    after suggestion (fixed to user-specified values).
    """
    if mode == 'focused':
        config = suggest_params(trial, FOCUSED_SEARCH_SPACE, FOCUSED_FIXED_PARAMS, mode)
    else:
        config = suggest_params(trial, FULL_SEARCH_SPACE, {}, mode)

    # Quota Mode: override entry_type with the forced value
    if forced_entry_type is not None:
        config['entry_type'] = forced_entry_type

    # Lock parameters: override with user-specified fixed values
    if locked_params:
        config.update(locked_params)

    # --- Exclusion check ---
    if excluded_fps is not None:
        fp = _config_fingerprint(config)
        if _is_too_similar(fp, excluded_fps, config, excluded_configs or []):
            trial.set_user_attr('excluded', True)
            trial.set_user_attr('config', config)
            return -998.0  # penalise but distinct from backtest failure (-999)

    # Run backtest
    try:
        trades = run_backtest(df_is, config, instrument)
    except Exception as e:
        warnings.warn(f"Backtest failed for trial {trial.number}: {e}")
        return -999.0

    # Compute metrics
    metrics = compute_metrics(trades)

    if metrics['n_trades'] < MIN_TRADES:
        return -999.0

    # Composite score
    score = compute_composite_score(metrics)

    # Robustness bonus for "wide" parameters
    pmax_length = config.get('pmax_length', 10)
    pmax_mult = config.get('pmax_multiplier', 3.0)
    if pmax_length >= 15 and pmax_mult >= 3.0:
        score *= ROBUSTNESS_BONUS

    # Store metrics in trial
    trial.set_user_attr('config', config)
    trial.set_user_attr('metrics', metrics)
    trial.set_user_attr('n_trades', metrics['n_trades'])
    trial.set_user_attr('calmar', metrics['calmar'])
    trial.set_user_attr('max_dd', metrics['max_dd'])
    trial.set_user_attr('win_rate', metrics['win_rate'])
    trial.set_user_attr('profit_factor', metrics['profit_factor'])

    return score


# ==============================================================================
# MAIN OPTIMIZATION
# ==============================================================================

def run_optimization(df: pd.DataFrame, instrument: str = 'UK100',
                     n_trials: int = 500, mode: str = 'focused',
                     is_ratio: float = 0.70, seed: int = 42,
                     progress_callback=None,
                     exclude_db_configs: bool = False,
                     selected_entries: list = None,
                     entry_quota: bool = False,
                     locked_params: dict = None,
                     data_file: str = '') -> dict:
    """
    Run the full Optuna optimization.

    Parameters:
        df: Full OHLCV DataFrame
        instrument: for transaction costs
        n_trials: number of optimization trials
        mode: 'focused' or 'full'
        is_ratio: in-sample fraction
        seed: random seed
        progress_callback: callable(trial_num, n_trials, best_score)
        exclude_db_configs: if True, penalise trials whose parameter
            combination already exists in the backtest database for
            this instrument AND data file (forces exploration of new regions)
        selected_entries: list of entry_type strings to explore
            (None = all SAFE_ENTRY_TYPES)
        entry_quota: if True, split trials equally across selected entries
            (prevents NSGA-II sampling bias towards dominant signals)
        locked_params: dict of {param_name: value} to lock during optimization.
            Locked params are fixed to specified values and NOT optimized.
        data_file: path to the data file (used to scope DB exclusion
            to the same instrument + timeframe combination)

    Returns:
        dict with:
            'study': optuna.Study
            'top_configs': list of dicts (top 20, sorted by score)
            'is_range': (start_date, end_date)
            'oos_range': (start_date, end_date)
            'df_is': DataFrame (in-sample)
            'df_oos': DataFrame (out-of-sample)
            'duration_sec': float
    """
    start_time = time.time()

    # Validate inputs to prevent OSError / ValueError downstream
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty. Please upload valid OHLCV data.")
    if not (0.1 <= is_ratio <= 0.95):
        raise ValueError(f"is_ratio must be between 0.1 and 0.95, got {is_ratio}")

    # Drop rows with NaN in OHLCV to prevent NaN propagation
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in DataFrame")
    df = df.dropna(subset=required_cols)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} valid bars after dropping NaN. Need >= 100.")

    # Split IS/OOS
    n = len(df)
    split_idx = int(n * is_ratio)
    if split_idx < 50 or (n - split_idx) < 50:
        raise ValueError(f"IS ({split_idx} bars) or OOS ({n - split_idx} bars) too small. Need >= 50 each.")

    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()

    is_range = (str(df_is.index[0]), str(df_is.index[-1]))
    oos_range = (str(df_oos.index[0]), str(df_oos.index[-1]))

    n_cpus = os.cpu_count() or 1
    # Use all CPUs for post-processing (ProcessPool = real parallelism)
    # Cap to avoid resource exhaustion in containers
    n_workers = max(1, min(n_cpus, 8))

    # --- Build exclusion set from DB ---
    excluded_fps = None
    excluded_configs = None
    n_excluded = 0
    if exclude_db_configs:
        try:
            from .backtest_db import BacktestDB
            db = BacktestDB()
            explored = db.get_explored_configs(instrument, data_file=data_file, keys=_FINGERPRINT_KEYS)
            excluded_fps = {c['_fingerprint'] for c in explored}
            excluded_configs = explored
            n_excluded = len(excluded_fps)
        except Exception as e:
            warnings.warn(f"Could not load DB configs for exclusion: {e}")
            excluded_fps = set()
            excluded_configs = []

    # --- Apply entry selection to search spaces ---
    from .config import SAFE_ENTRY_TYPES
    if selected_entries is None or len(selected_entries) == 0:
        selected_entries = list(SAFE_ENTRY_TYPES)
    # Filter to only valid entries
    selected_entries = [e for e in selected_entries if e in SAFE_ENTRY_TYPES]
    if not selected_entries:
        selected_entries = list(SAFE_ENTRY_TYPES)

    # Override the entry_type choices in the search spaces
    # We patch module-level dicts temporarily — safe because
    # the optimizer runs in a controlled single-thread context
    _orig_focused_entries = FOCUSED_SEARCH_SPACE['entry_type']['choices']
    _orig_full_entries = FULL_SEARCH_SPACE['entry_type']['choices']
    FOCUSED_SEARCH_SPACE['entry_type']['choices'] = selected_entries
    FULL_SEARCH_SPACE['entry_type']['choices'] = selected_entries

    print(f"\n{'='*60}")
    print(f"PF AI Lab 5.0 — Optimizer Mode: {mode.upper()}")
    print(f"{'='*60}")
    print(f"  IS period:  {is_range[0]} -> {is_range[1]} ({len(df_is)} bars)")
    print(f"  OOS period: {oos_range[0]} -> {oos_range[1]} ({len(df_oos)} bars)")
    print(f"  Instrument: {instrument}")
    print(f"  Trials:     {n_trials}")
    print(f"  CPUs:       {n_cpus} (ProcessPool for post-processing)")
    print(f"  Entries:    {', '.join(selected_entries)} ({len(selected_entries)} signals)")
    if entry_quota:
        print(f"  Quota Mode: ON — ~{n_trials // len(selected_entries)} trials/signal (equal distribution)")
    if exclude_db_configs:
        print(f"  Exclude DB: {n_excluded} unique configs from {instrument} will be penalised")
    if locked_params:
        print(f"  Locked:     {len(locked_params)} param(s) — {', '.join(f'{k}={v}' for k,v in locked_params.items())}")
    print(f"{'='*60}\n")

    # Create sampler
    sampler = optuna.samplers.NSGAIISampler(seed=seed)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f'pfalgo_v691_{instrument}_{mode}'
    )

    def _callback(study, trial):
        if progress_callback:
            progress_callback(trial.number + 1, n_trials, _safe_best_value(study))
        if (trial.number + 1) % 50 == 0:
            print(f"  Trial {trial.number + 1}/{n_trials} | "
                  f"Best Score: {_safe_best_value(study):.4f} | "
                  f"Trades: {trial.user_attrs.get('n_trades', '?')} | "
                  f"Calmar: {trial.user_attrs.get('calmar', '?')}")

    # ================================================================
    # PARALLEL OPTIMIZATION via multiprocessing
    # ================================================================
    # Optuna n_jobs>1 uses THREADS → GIL blocks → only 1 core used.
    # Solution: split trials across PROCESSES via ProcessPoolExecutor.
    # Each worker gets its own Python interpreter → true parallelism.
    #
    # Strategy: divide n_trials into n_workers batches, each worker
    # creates its own study with the same sampler (different seed) and
    # runs its batch.  Results are merged into the main study.
    # ================================================================
    use_parallel_trials = n_workers >= 2 and n_trials >= 40

    if use_parallel_trials:
        # Quota mode: create one batch per entry_type, then split each
        # batch across CPUs.  Non-quota: split total trials across CPUs.
        if entry_quota and len(selected_entries) > 1:
            # ── QUOTA MODE: equal trials per entry type ──
            trials_per_entry = n_trials // len(selected_entries)
            extra = n_trials % len(selected_entries)
            worker_args = []
            trials_per_worker = []

            print(f"  Strategy: Quota Mode — {len(selected_entries)} signals × "
                  f"~{trials_per_entry} trials, {n_workers} CPU workers")

            # For each entry type, create 1+ workers
            # If we have 3 entries and 8 CPUs: 2-3 workers per entry
            workers_per_entry = max(1, n_workers // len(selected_entries))

            for eidx, entry_name in enumerate(selected_entries):
                e_trials = trials_per_entry + (1 if eidx < extra else 0)
                # Split this entry's trials across its assigned workers
                e_batch = e_trials // workers_per_entry
                e_remainder = e_trials % workers_per_entry
                for w in range(workers_per_entry):
                    w_trials = e_batch + (1 if w < e_remainder else 0)
                    if w_trials <= 0:
                        continue
                    w_seed = seed + eidx * 100 + w
                    worker_args.append((
                        df_is, instrument, mode, w_trials, w_seed,
                        excluded_fps, excluded_configs,
                        entry_name,  # forced_entry_type
                        selected_entries,  # entry_choices (not used when forced)
                        locked_params,  # locked parameters
                    ))
                    trials_per_worker.append(w_trials)
        else:
            # ── STANDARD MODE: split total trials across CPUs ──
            print(f"  Strategy: {n_workers} parallel workers × ~{n_trials // n_workers} trials each")
            batch_size = n_trials // n_workers
            remainder = n_trials % n_workers

            worker_args = []
            trials_per_worker = []
            for w in range(n_workers):
                w_trials = batch_size + (1 if w < remainder else 0)
                w_seed = seed + w
                worker_args.append((
                    df_is, instrument, mode, w_trials, w_seed,
                    excluded_fps, excluded_configs,
                    None,  # forced_entry_type = None (Optuna picks)
                    selected_entries,  # entry_choices
                    locked_params,  # locked parameters
                ))
                trials_per_worker.append(w_trials)

        # Run workers in parallel — progress updated per worker completion
        all_configs_from_workers = []
        completed_trials_count = 0
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                future_map = {pool.submit(_parallel_optuna_worker, a): i
                              for i, a in enumerate(worker_args)}

                for fut in as_completed(future_map):
                    widx = future_map[fut]
                    try:
                        worker_results = fut.result()
                        all_configs_from_workers.extend(worker_results)
                    except Exception as e:
                        warnings.warn(f"Worker {widx} failed: {e}")

                    # Progress: count trials from completed workers
                    completed_trials_count += trials_per_worker[widx]
                    if progress_callback:
                        progress_callback(completed_trials_count, n_trials, 0)
                    print(f"    Worker {widx+1}/{n_workers} done "
                          f"({trials_per_worker[widx]} trials, "
                          f"{len(worker_results) if 'worker_results' in dir() else 0} valid)")

        except (OSError, RuntimeError) as pool_err:
            warnings.warn(f"Parallel trials failed ({pool_err}), falling back to sequential")
            use_parallel_trials = False

    if not use_parallel_trials:
        # Fallback: sequential (single-core)
        if entry_quota and len(selected_entries) > 1:
            # Sequential Quota: run each entry type in turn
            trials_per_entry = n_trials // len(selected_entries)
            extra = n_trials % len(selected_entries)
            print(f"  Strategy: sequential Quota — {len(selected_entries)} signals × ~{trials_per_entry} trials")
            global_trial_count = [0]
            for eidx, entry_name in enumerate(selected_entries):
                e_trials = trials_per_entry + (1 if eidx < extra else 0)
                print(f"    [{eidx+1}/{len(selected_entries)}] {entry_name}: {e_trials} trials")
                def _quota_callback(study_inner, trial_inner, _eidx=eidx, _name=entry_name):
                    global_trial_count[0] += 1
                    if progress_callback:
                        progress_callback(global_trial_count[0], n_trials, _safe_best_value(study))
                study.optimize(
                    lambda trial, _et=entry_name: objective(
                        trial, df_is, instrument, mode,
                        excluded_fps=excluded_fps,
                        excluded_configs=excluded_configs,
                        forced_entry_type=_et,
                        locked_params=locked_params),
                    n_trials=e_trials,
                    callbacks=[_quota_callback],
                    show_progress_bar=False,
                    n_jobs=1,
                )
        else:
            print(f"  Strategy: sequential (1 core)")
            study.optimize(
                lambda trial: objective(trial, df_is, instrument, mode,
                                        excluded_fps=excluded_fps,
                                        excluded_configs=excluded_configs,
                                        locked_params=locked_params),
                n_trials=n_trials,
                callbacks=[_callback],
                show_progress_bar=False,
                n_jobs=1,
            )
    else:
        # Inject worker results into the main study as completed trials
        # by running them through the study so it tracks best_value etc.
        for wc in all_configs_from_workers:
            cfg = wc['config']
            score = wc['score']
            metrics = wc.get('metrics', {})
            # Use FixedTrial to replay the exact config
            dist = {}
            try:
                trial = study.ask()
                trial.set_user_attr('config', cfg)
                trial.set_user_attr('metrics', metrics)
                trial.set_user_attr('n_trades', metrics.get('n_trades', 0))
                trial.set_user_attr('calmar', metrics.get('calmar', 0))
                trial.set_user_attr('max_dd', metrics.get('max_dd', 0))
                trial.set_user_attr('win_rate', metrics.get('win_rate', 0))
                trial.set_user_attr('profit_factor', metrics.get('profit_factor', 0))
                study.tell(trial, score)
            except Exception:
                pass

    # Restore original search space entry_type choices
    FOCUSED_SEARCH_SPACE['entry_type']['choices'] = _orig_focused_entries
    FULL_SEARCH_SPACE['entry_type']['choices'] = _orig_full_entries

    # Extract top configs
    # Count excluded trials for reporting
    n_excluded_trials = sum(1 for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                            and t.user_attrs.get('excluded'))

    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                        and t.value is not None and t.value > -900]

    completed_trials.sort(key=lambda t: t.value, reverse=True)

    # ==========================================================================
    # POST-PROCESSING with ProcessPoolExecutor (true multi-core parallelism)
    # Benchmarked: ThreadPool=10.9s, ProcessPool=1.2s for 50 backtests (8.9x faster)
    # ==========================================================================

    # --- OOS validation on top 20 ---
    t_post = time.time()
    print(f"\n  Post-processing with {n_workers} CPU workers (ProcessPool)...")

    oos_candidates = []
    for t in completed_trials[:20]:
        oos_candidates.append({
            'config': t.user_attrs.get('config', {}),
            'metrics_is': t.user_attrs.get('metrics', {}),
            'score': t.value,
            'trial_number': t.number,
        })

    # --- Phase 1: OOS backtests (parallel) ---
    n_oos = len(oos_candidates)
    # Progress total: n_trials + OOS(20) + perturbation(20) + WF(20) = n_trials + 60
    total_progress = n_trials + n_oos * 3  # 3 phases, each up to 20
    print(f"    OOS validation on {n_oos} configs...")
    oos_args = [(c['config'], df_oos, instrument) for c in oos_candidates]

    oos_results = [None] * len(oos_candidates)
    if oos_args:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                future_map = {pool.submit(_run_oos_backtest_worker, a): i
                              for i, a in enumerate(oos_args)}
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        oos_results[idx] = fut.result()
                    except Exception:
                        oos_results[idx] = {'n_trades': 0, 'calmar': 0, 'profit_factor': 0, 'max_dd': 0}
                    if progress_callback:
                        progress_callback(n_trials + idx + 1, total_progress, _safe_best_value(study))
        except (OSError, RuntimeError) as pool_err:
            # Fallback to sequential execution if ProcessPool fails
            # (can happen in containers with limited /dev/shm or fork restrictions)
            warnings.warn(f"ProcessPool failed ({pool_err}), falling back to sequential OOS")
            for i, args in enumerate(oos_args):
                try:
                    oos_results[i] = _run_oos_backtest_worker(args)
                except Exception:
                    oos_results[i] = {'n_trades': 0, 'calmar': 0, 'profit_factor': 0, 'max_dd': 0}
                if progress_callback:
                    progress_callback(n_trials + i + 1, total_progress, _safe_best_value(study))

    top_configs = []
    for i, c in enumerate(oos_candidates):
        top_configs.append({
            'rank': i + 1,
            'score': c['score'],
            'config': c['config'],
            'metrics_is': c['metrics_is'],
            'metrics_oos': oos_results[i],
            'trial_number': c['trial_number'],
        })

    # --- Phase 2: Perturbation robustness on ALL 20 (parallel) ---
    n_perturb = len(top_configs)
    if n_perturb > 0:
        print(f"    ±{PERTURBATION_PCT}% perturbation on all {n_perturb} configs...")
        perturb_args = [(tc['config'], df_is, instrument) for tc in top_configs[:n_perturb]]

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                future_map = {pool.submit(_perturbation_worker, a): i
                              for i, a in enumerate(perturb_args)}
                done_count = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        top_configs[idx]['perturbation_pass'] = fut.result()
                    except Exception:
                        top_configs[idx]['perturbation_pass'] = {'stable': False, 'degradation': 1.0, 'n_surviving': 0}
                    done_count += 1
                    if progress_callback:
                        progress_callback(n_trials + n_oos + done_count, total_progress, _safe_best_value(study))
        except (OSError, RuntimeError) as pool_err:
            warnings.warn(f"ProcessPool failed ({pool_err}), falling back to sequential perturbation")
            for i, args in enumerate(perturb_args):
                try:
                    top_configs[i]['perturbation_pass'] = _perturbation_worker(args)
                except Exception:
                    top_configs[i]['perturbation_pass'] = {'stable': False, 'degradation': 1.0, 'n_surviving': 0}
                if progress_callback:
                    progress_callback(n_trials + n_oos + i + 1, total_progress, _safe_best_value(study))

    # --- Phase 3: Walk-Forward analysis on ALL 20 (parallel) ---
    n_wf = len(top_configs)
    _default_wf = {
        'stability_score': 0, 'consistency': 0,
        'avg_calmar': 0, 'avg_pf': 0,
        'degradation': False, 'n_windows': 0,
        'windows': [],
    }

    def _process_wf_result(tc, wf_result):
        tc['walk_forward'] = {
            'stability_score': wf_result.get('stability_score', 0),
            'consistency': wf_result.get('consistency', 0),
            'avg_calmar': wf_result.get('avg_calmar', 0),
            'avg_pf': wf_result.get('avg_pf', 0),
            'degradation': wf_result.get('degradation', False),
            'n_windows': len(wf_result.get('windows', [])),
            'windows': wf_result.get('windows', []),
        }
        print(f"      Rank #{tc['rank']}: WF={wf_result.get('stability_score', 0):.0f}/100 | "
              f"Cons={wf_result.get('consistency', 0):.0f}% | "
              f"Degr={'YES' if wf_result.get('degradation') else 'NO'}")

    if n_wf > 0:
        print(f"    Walk-Forward on all {n_wf} configs...")
        wf_args = [(tc['config'], df, instrument) for tc in top_configs[:n_wf]]

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                future_map = {pool.submit(_walk_forward_worker, a): i
                              for i, a in enumerate(wf_args)}
                wf_done = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    tc = top_configs[idx]
                    try:
                        wf_result = fut.result()
                        _process_wf_result(tc, wf_result)
                    except Exception as e:
                        tc['walk_forward'] = {**_default_wf, 'error': str(e)}
                    wf_done += 1
                    if progress_callback:
                        progress_callback(n_trials + n_oos + n_perturb + wf_done, total_progress, _safe_best_value(study))
        except (OSError, RuntimeError) as pool_err:
            warnings.warn(f"ProcessPool failed ({pool_err}), falling back to sequential WF")
            for i, args in enumerate(wf_args):
                tc = top_configs[i]
                try:
                    wf_result = _walk_forward_worker(args)
                    _process_wf_result(tc, wf_result)
                except Exception as e:
                    tc['walk_forward'] = {**_default_wf, 'error': str(e)}
                if progress_callback:
                    progress_callback(n_trials + n_oos + n_perturb + i + 1, total_progress, _safe_best_value(study))

    t_post_elapsed = time.time() - t_post
    duration = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Optimization complete in {duration:.1f}s")
    print(f"  Trials: {duration - t_post_elapsed:.1f}s | Post-processing: {t_post_elapsed:.1f}s")
    print(f"  Top Score: {_safe_best_value(study):.4f}")
    print(f"  Valid trials: {len(completed_trials)}/{n_trials}")
    if n_excluded_trials > 0:
        print(f"  Excluded (DB duplicate): {n_excluded_trials} trials skipped")
    print(f"{'='*60}\n")

    return {
        'study': study,
        'top_configs': top_configs,
        'is_range': is_range,
        'oos_range': oos_range,
        'df_is': df_is,
        'df_oos': df_oos,
        'duration_sec': duration,
        'instrument': instrument,
        'n_excluded_trials': n_excluded_trials,
    }


# ==============================================================================
# PERTURBATION ROBUSTNESS (called inside ProcessPool workers)
# ==============================================================================

def _check_perturbation_robustness(config: dict, df: pd.DataFrame,
                                     instrument: str) -> dict:
    """
    Perturb numeric parameters +/-10% and check that performance
    doesn't degrade catastrophically. Returns stability score.
    """
    rng = np.random.RandomState(123)

    # Get baseline
    try:
        base_trades = run_backtest(df, config, instrument)
        base_metrics = compute_metrics(base_trades)
        base_calmar = base_metrics.get('calmar', 0)
        base_pf = base_metrics.get('profit_factor', 0)
    except Exception:
        return {'stable': False, 'degradation': 1.0, 'n_surviving': 0}

    if base_calmar <= 0:
        return {'stable': False, 'degradation': 1.0, 'n_surviving': 0}

    numeric_keys = ['pmax_length', 'pmax_multiplier', 'pmax_ma_factor', 'pmax_ma_distance',
                    'sig_ma_fast', 'sig_ma_slow',
                    'don_length',
                    'rsi_div_length', 'rsi_div_pivot',
                    'lt_length', 'lt_multiplier',
                    'mt_length', 'mt_multiplier',
                    'diff_ma_red_pct', 'order_pause',
                    'fractal_buffer', 'length_bb_ma', 'mult_bb_ma', 'length_bb_sma',
                    # V5.2 numeric params
                    'sl_pct', 'atr_sl_mult', 'sl_vol_min_mult', 'sl_vol_max_mult',
                    'tp_pct', 'tp_sl_ratio',
                    'be_trigger_pct', 'tsl_pct',
                    'max_hold_days', 'adx_trend_threshold', 'vol_percentile_threshold',
                    # V5.3 Tier 3 numeric params
                    'tp1_pct', 'tp1_size', 'tp2_pct', 'tp2_size',
                    'session1_start', 'session1_end',
                    'vix_threshold_high',
                    'macro_min', 'macro_max',
                    'qq_length']

    surviving = 0
    calmars = []

    for _ in range(PERTURBATION_N_RUNS):
        perturbed = config.copy()
        for key in numeric_keys:
            if key in perturbed and isinstance(perturbed[key], (int, float)):
                val = perturbed[key]
                delta = val * PERTURBATION_PCT / 100.0
                new_val = val + rng.uniform(-delta, delta)
                if isinstance(val, int):
                    perturbed[key] = max(1, int(round(new_val)))
                else:
                    perturbed[key] = round(max(0.1, new_val), 2)
        try:
            trades = run_backtest(df, perturbed, instrument)
            m = compute_metrics(trades)
            c = m.get('calmar', 0)
            calmars.append(c)
            if c > base_calmar * 0.5 and m['n_trades'] >= 20:
                surviving += 1
        except Exception:
            calmars.append(0)

    avg_calmar = np.mean(calmars) if calmars else 0
    degradation = 1 - (avg_calmar / base_calmar) if base_calmar > 0 else 1.0

    return {
        'stable': surviving >= PERTURBATION_N_RUNS * 0.6,
        'degradation': round(max(0, min(1, degradation)), 3),
        'n_surviving': surviving,
        'n_total': PERTURBATION_N_RUNS,
        'avg_calmar': round(avg_calmar, 3),
        'base_calmar': round(base_calmar, 3),
    }
