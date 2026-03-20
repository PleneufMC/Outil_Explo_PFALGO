"""
PF AI Lab 5.0.4 — Optimiseur NSGA-II multi-objectifs.

Utilise Optuna avec TPESampler ou NSGAIISampler pour optimiser
les parametres de la strategie P.F.Algo.

Objectifs (mode NSGA-II):
  1. Maximiser Profit Factor
  2. Minimiser Max Drawdown
  3. Maximiser Sharpe Ratio

Contraintes cross-engine:
  - Penalise les zones sensibles (pmax_length < 10, multiplier < 2.5)
  - Verifie le ratio params/trades < 0.15
  - Prefere les parametres dans la zone robuste (cross-engine reproducible)

Workflow hybride recommande:
  Python EXPLORER (NSGA-II, SAFE entries, +/-10-30% disclaimer)
    -> Top 5-10 configs
  TradingView VALIDATOR (Strategy Tester = source de verite)
    -> Export real trades
  Python ANALYZER (Monte Carlo, Walk-Forward sur vrais trades TV)
    -> DEPLOY / MONITOR verdict
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import warnings
import time

from pineguard.config import PineGuardConfig, validate_config, SENSITIVITY_MATRIX


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_trials: List[Dict]
    pareto_front: List[Dict]
    n_trials: int
    duration_seconds: float
    search_space: Dict
    warnings: List[str]


def _build_config_from_trial(trial, search_space: Dict, base_config: PineGuardConfig) -> PineGuardConfig:
    """Build a PineGuardConfig from an Optuna trial's suggested parameters."""
    params = {}
    for name, space in search_space.items():
        if space['type'] == 'int':
            params[name] = trial.suggest_int(name, space['low'], space['high'],
                                             step=space.get('step', 1))
        elif space['type'] == 'float':
            params[name] = trial.suggest_float(name, space['low'], space['high'],
                                               step=space.get('step', None))
        elif space['type'] == 'categorical':
            params[name] = trial.suggest_categorical(name, space['choices'])

    # Merge with base config
    config_dict = base_config.to_dict()
    config_dict.update(params)
    return PineGuardConfig.from_dict(config_dict)


def _sensitivity_penalty(config: PineGuardConfig) -> float:
    """
    Penalite [0, 1] pour parametres en zone sensible.
    0 = tout robuste, 1 = tout sensible.
    """
    penalties = 0
    total_checks = 4

    if config.pmax_length < 10:
        penalties += 1
    if config.pmax_multiplier < 2.5:
        penalties += 1
    if config.don_length < 15:
        penalties += 1
    if config.sl_pct is not None and config.sl_pct < 2.0:
        penalties += 1

    return penalties / total_checks


def get_default_search_space(entry_type: str = 'Donc+Chikou') -> Dict:
    """
    Espace de recherche par defaut pour un type d'entree donne.

    Prefere la zone robuste pour la reproductibilite cross-engine.
    """
    # Base PMax params (toujours optimises)
    space = {
        'pmax_length': {'type': 'int', 'low': 8, 'high': 25, 'step': 1},
        'pmax_multiplier': {'type': 'float', 'low': 2.0, 'high': 5.0, 'step': 0.1},
        'pmax_ma_type': {'type': 'categorical', 'choices': ['EMA', 'SMA', 'RMA', 'DEMA', 'TEMA']},
    }

    # Entry-specific params
    if entry_type == 'Donc+Chikou':
        space.update({
            'don_length': {'type': 'int', 'low': 15, 'high': 40, 'step': 1},
            'chikou_shift': {'type': 'int', 'low': 20, 'high': 35, 'step': 1},
        })
    elif entry_type == 'Boll+SMA':
        space.update({
            'boll_length': {'type': 'int', 'low': 15, 'high': 30, 'step': 1},
            'boll_mult': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.1},
            'boll_sma_length': {'type': 'int', 'low': 30, 'high': 100, 'step': 5},
        })
    elif entry_type == 'RSI Divergence':
        space.update({
            'rsi_div_length': {'type': 'int', 'low': 10, 'high': 21, 'step': 1},
            'rsi_div_lookback': {'type': 'int', 'low': 3, 'high': 10, 'step': 1},
        })
    elif entry_type == 'MM Cross':
        space.update({
            'mm_cross_ma1_type': {'type': 'categorical', 'choices': ['EMA', 'TRIMA']},
            'mm_cross_ma1_len': {'type': 'int', 'low': 5, 'high': 50, 'step': 1},
            'mm_cross_ma2_type': {'type': 'categorical', 'choices': ['EMA', 'TRIMA']},
            'mm_cross_ma2_len': {'type': 'int', 'low': 10, 'high': 200, 'step': 1},
        })

    # SL params
    space['sl_pct'] = {'type': 'float', 'low': 2.0, 'high': 10.0, 'step': 0.5}

    return space


def optimize_single_objective(
    df: pd.DataFrame,
    objective_metric: str = 'sharpe_ratio',
    entry_type: str = 'Donc+Chikou',
    search_space: Optional[Dict] = None,
    n_trials: int = 100,
    base_config: Optional[PineGuardConfig] = None,
    min_trades: int = 10,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimisation mono-objectif avec Optuna TPE.

    Args:
        df: DataFrame OHLCV
        objective_metric: Metrique a maximiser (sharpe_ratio, profit_factor, calmar_ratio, etc.)
        entry_type: Type d'entree
        search_space: Espace de recherche (None = defaut)
        n_trials: Nombre d'essais
        base_config: Configuration de base
        min_trades: Nombre minimum de trades (penalise si insuffisant)
        verbose: Afficher progression

    Returns:
        OptimizationResult
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna requis pour l'optimisation. "
            "Installer avec: pip install optuna"
        )

    from pf_ai_lab.runner import run_backtest

    if search_space is None:
        search_space = get_default_search_space(entry_type)

    if base_config is None:
        base_config = PineGuardConfig(entry_type=entry_type)

    start_time = time.time()
    all_trials = []

    def objective(trial):
        config = _build_config_from_trial(trial, search_space, base_config)
        result = run_backtest(df=df.copy(), config=config, verbose=False)

        if 'error' in result:
            return float('-inf')

        metrics = result['metrics']
        n_trades = metrics['total_trades']

        # Penalite si pas assez de trades
        if n_trades < min_trades:
            return float('-inf')

        # Penalite zone sensible
        penalty = _sensitivity_penalty(config)
        raw_value = metrics.get(objective_metric, 0)

        # Ajustement: penalite proportionnelle
        adjusted = raw_value * (1 - 0.2 * penalty)

        trial_info = {
            'params': config.to_dict(),
            'metrics': metrics,
            'sensitivity_penalty': penalty,
            'raw_value': raw_value,
            'adjusted_value': adjusted,
        }
        all_trials.append(trial_info)

        return adjusted

    # Suppress Optuna logging unless verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    duration = time.time() - start_time

    best = study.best_trial
    best_trial_info = all_trials[best.number] if best.number < len(all_trials) else {}

    return OptimizationResult(
        best_params=best.params,
        best_metrics=best_trial_info.get('metrics', {}),
        all_trials=all_trials,
        pareto_front=[best_trial_info],
        n_trials=n_trials,
        duration_seconds=round(duration, 1),
        search_space=search_space,
        warnings=_generate_optimization_warnings(best.params, search_space),
    )


def optimize_multi_objective(
    df: pd.DataFrame,
    entry_type: str = 'Donc+Chikou',
    search_space: Optional[Dict] = None,
    n_trials: int = 200,
    base_config: Optional[PineGuardConfig] = None,
    min_trades: int = 10,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimisation NSGA-II multi-objectifs.

    Objectifs:
      1. Maximiser Profit Factor
      2. Minimiser Max Drawdown (inverser pour maximiser)
      3. Maximiser Sharpe Ratio

    Retourne le front de Pareto.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna requis. pip install optuna")

    from pf_ai_lab.runner import run_backtest

    if search_space is None:
        search_space = get_default_search_space(entry_type)

    if base_config is None:
        base_config = PineGuardConfig(entry_type=entry_type)

    start_time = time.time()
    all_trials = []

    def objective(trial):
        config = _build_config_from_trial(trial, search_space, base_config)
        result = run_backtest(df=df.copy(), config=config, verbose=False)

        if 'error' in result:
            return float('-inf'), float('-inf'), float('-inf')

        metrics = result['metrics']

        if metrics['total_trades'] < min_trades:
            return float('-inf'), float('-inf'), float('-inf')

        penalty = _sensitivity_penalty(config)

        pf = metrics.get('profit_factor', 0) * (1 - 0.15 * penalty)
        neg_dd = -metrics.get('max_drawdown_pct', 100)  # Minimiser DD = maximiser -DD
        sharpe = metrics.get('sharpe_ratio', 0) * (1 - 0.15 * penalty)

        trial_info = {
            'params': config.to_dict(),
            'metrics': metrics,
            'objectives': {'profit_factor': pf, 'neg_max_dd': neg_dd, 'sharpe': sharpe},
        }
        all_trials.append(trial_info)

        return pf, neg_dd, sharpe

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        directions=['maximize', 'maximize', 'maximize'],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    duration = time.time() - start_time

    # Extract Pareto front
    pareto = []
    for trial in study.best_trials:
        if trial.number < len(all_trials):
            pareto.append(all_trials[trial.number])

    # Best by Sharpe on the Pareto front
    best_params = {}
    best_metrics = {}
    if pareto:
        best_by_sharpe = max(pareto, key=lambda t: t['objectives'].get('sharpe', 0))
        best_params = best_by_sharpe.get('params', {})
        best_metrics = best_by_sharpe.get('metrics', {})

    return OptimizationResult(
        best_params=best_params,
        best_metrics=best_metrics,
        all_trials=all_trials,
        pareto_front=pareto,
        n_trials=n_trials,
        duration_seconds=round(duration, 1),
        search_space=search_space,
        warnings=_generate_optimization_warnings(best_params, search_space),
    )


def _generate_optimization_warnings(params: Dict, search_space: Dict) -> List[str]:
    """Generate warnings about optimization results."""
    warnings_list = []

    pmax_len = params.get('pmax_length', 10)
    pmax_mult = params.get('pmax_multiplier', 3.0)
    don_len = params.get('don_length', 20)
    sl_pct = params.get('sl_pct', 5.0)

    if pmax_len < 10:
        warnings_list.append(
            f"pmax_length={pmax_len} en zone sensible (< 10). "
            f"Risque de divergence Pine/Python."
        )
    if pmax_mult < 2.5:
        warnings_list.append(
            f"pmax_multiplier={pmax_mult} en zone sensible (< 2.5). "
            f"Crossovers instables."
        )
    if don_len < 15:
        warnings_list.append(
            f"don_length={don_len} en zone sensible (< 15). "
            f"Canal Donchian volatil."
        )
    if sl_pct < 2.0:
        warnings_list.append(
            f"sl_pct={sl_pct}% en zone sensible (< 2%). "
            f"SL trop serre, hits par micro-ecarts."
        )

    # Check if best params are at boundary
    for name, space in search_space.items():
        val = params.get(name)
        if val is not None and space['type'] in ('int', 'float'):
            if val == space['low'] or val == space['high']:
                warnings_list.append(
                    f"{name}={val} est a la frontiere de l'espace de recherche "
                    f"[{space['low']}, {space['high']}]. Elargir?"
                )

    return warnings_list
