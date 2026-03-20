"""
PF AI Lab 5.0.4 — Walk-Forward Analysis (WFA).

Methode anti-overfitting: decoupe les donnees en fenetres In-Sample / Out-of-Sample.
Pour chaque fenetre:
  1. Optimise sur IS (In-Sample)
  2. Teste les meilleurs params sur OOS (Out-of-Sample)
  3. Verifie la degradation IS -> OOS

Criteres de validation:
  - WFE (Walk-Forward Efficiency) = OOS_metric / IS_metric > 0.5
  - Stabilite: ecart-type des WFE < 0.3
  - Pas de degradation monotone
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from pineguard.config import PineGuardConfig


@dataclass
class WFAWindow:
    """Single Walk-Forward window result."""
    window_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_bars: int
    oos_bars: int
    best_params: Dict[str, Any]
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    wfe: float  # Walk-Forward Efficiency


@dataclass
class WFAResult:
    """Walk-Forward Analysis result."""
    windows: List[WFAWindow]
    avg_wfe: float
    std_wfe: float
    is_valid: bool
    validation_messages: List[str]
    aggregate_oos_metrics: Dict[str, float]
    monotone_degradation: bool


def run_walk_forward(
    df: pd.DataFrame,
    config: PineGuardConfig,
    n_windows: int = 5,
    is_ratio: float = 0.7,
    objective_metric: str = 'sharpe_ratio',
    n_opt_trials: int = 50,
    min_trades_per_window: int = 5,
    verbose: bool = True,
) -> WFAResult:
    """
    Execute une Walk-Forward Analysis.

    Args:
        df: DataFrame OHLCV complet
        config: Configuration de base
        n_windows: Nombre de fenetres WFA (default 5)
        is_ratio: Ratio In-Sample / fenetre totale (default 0.7)
        objective_metric: Metrique pour l'optimisation IS
        n_opt_trials: Trials Optuna par fenetre IS
        min_trades_per_window: Minimum trades pour valider une fenetre
        verbose: Afficher progression

    Returns:
        WFAResult
    """
    from pf_ai_lab.runner import run_backtest
    from pf_ai_lab.optimizer import optimize_single_objective, get_default_search_space

    n_total = len(df)
    window_size = n_total // n_windows
    is_size = int(window_size * is_ratio)
    oos_size = window_size - is_size

    if verbose:
        print(f"\n{'='*60}")
        print(f"Walk-Forward Analysis: {n_windows} fenetres")
        print(f"  Total: {n_total} barres | IS: {is_size} | OOS: {oos_size}")
        print(f"  Objectif: {objective_metric} | Trials/fenetre: {n_opt_trials}")
        print(f"{'='*60}")

    windows: List[WFAWindow] = []
    search_space = get_default_search_space(config.entry_type)

    for w in range(n_windows):
        start_idx = w * window_size
        is_end_idx = start_idx + is_size
        oos_end_idx = min(start_idx + window_size, n_total)

        df_is = df.iloc[start_idx:is_end_idx].copy()
        df_oos = df.iloc[is_end_idx:oos_end_idx].copy()

        if len(df_is) < 50 or len(df_oos) < 20:
            if verbose:
                print(f"  Fenetre {w+1}: SKIP (pas assez de donnees)")
            continue

        if verbose:
            print(f"\n  Fenetre {w+1}/{n_windows}: "
                  f"IS [{df_is.index[0].strftime('%Y-%m-%d')} -> "
                  f"{df_is.index[-1].strftime('%Y-%m-%d')}] "
                  f"OOS [{df_oos.index[0].strftime('%Y-%m-%d')} -> "
                  f"{df_oos.index[-1].strftime('%Y-%m-%d')}]")

        # Optimize on IS
        try:
            opt_result = optimize_single_objective(
                df=df_is,
                objective_metric=objective_metric,
                entry_type=config.entry_type,
                search_space=search_space,
                n_trials=n_opt_trials,
                base_config=config,
                min_trades=min_trades_per_window,
                verbose=False,
            )
            best_params = opt_result.best_params
            is_metrics = opt_result.best_metrics
        except Exception as e:
            if verbose:
                print(f"    IS optimisation echouee: {e}")
            continue

        # Apply best params on OOS
        oos_config_dict = config.to_dict()
        oos_config_dict.update(best_params)
        oos_config = PineGuardConfig.from_dict(oos_config_dict)

        oos_result = run_backtest(df=df_oos, config=oos_config, verbose=False)
        if 'error' in oos_result:
            if verbose:
                print(f"    OOS backtest echoue")
            continue

        oos_metrics = oos_result['metrics']

        # WFE = OOS / IS metric
        is_val = is_metrics.get(objective_metric, 0)
        oos_val = oos_metrics.get(objective_metric, 0)
        wfe = oos_val / is_val if is_val != 0 else 0

        window = WFAWindow(
            window_id=w,
            is_start=df_is.index[0],
            is_end=df_is.index[-1],
            oos_start=df_oos.index[0],
            oos_end=df_oos.index[-1],
            is_bars=len(df_is),
            oos_bars=len(df_oos),
            best_params=best_params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            wfe=round(wfe, 3),
        )
        windows.append(window)

        if verbose:
            print(f"    IS {objective_metric}: {is_val:.3f} | "
                  f"OOS: {oos_val:.3f} | WFE: {wfe:.3f}")

    # Aggregate results
    if not windows:
        return WFAResult(
            windows=[], avg_wfe=0, std_wfe=0, is_valid=False,
            validation_messages=["Aucune fenetre valide"],
            aggregate_oos_metrics={}, monotone_degradation=False,
        )

    wfes = [w.wfe for w in windows]
    avg_wfe = np.mean(wfes)
    std_wfe = np.std(wfes)

    # Aggregate OOS metrics (mean across windows)
    agg_keys = ['total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio',
                'max_drawdown_pct', 'total_pnl']
    agg_oos = {}
    for key in agg_keys:
        vals = [w.oos_metrics.get(key, 0) for w in windows]
        agg_oos[key] = round(np.mean(vals), 3)

    # Check monotone degradation
    mono_degrad = all(wfes[i] > wfes[i+1] for i in range(len(wfes)-1)) if len(wfes) > 2 else False

    # Validation
    messages = []
    is_valid = True

    if avg_wfe < 0.5:
        messages.append(f"WFE moyen ({avg_wfe:.2f}) < 0.5 : possible overfitting")
        is_valid = False
    else:
        messages.append(f"WFE moyen ({avg_wfe:.2f}) >= 0.5 : OK")

    if std_wfe > 0.3:
        messages.append(f"Ecart-type WFE ({std_wfe:.2f}) > 0.3 : instable")
        is_valid = False
    else:
        messages.append(f"Ecart-type WFE ({std_wfe:.2f}) <= 0.3 : stable")

    if mono_degrad:
        messages.append("Degradation monotone detectee : suspecter regime change")
        is_valid = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"WFA Resume:")
        print(f"  WFE moyen: {avg_wfe:.3f} (std: {std_wfe:.3f})")
        print(f"  Validation: {'PASS' if is_valid else 'FAIL'}")
        for msg in messages:
            print(f"    {msg}")
        print(f"{'='*60}")

    return WFAResult(
        windows=windows,
        avg_wfe=round(avg_wfe, 3),
        std_wfe=round(std_wfe, 3),
        is_valid=is_valid,
        validation_messages=messages,
        aggregate_oos_metrics=agg_oos,
        monotone_degradation=mono_degrad,
    )
