"""
PF AI Lab 5.0.4 — Runner principal de backtest.

Pipeline complet:
  1. Chargement donnees (CSV MT5/TV ou sample)
  2. Calcul indicateurs (PMax, ATR, RSI, Bollinger, Donchian, ADX)
  3. Calcul signaux d'entree (7 types, V69.2: +MM Cross)
  4. Application filtres (entry_allowed, LT, MT, ADX regime)
  5. Execution backtest 3 phases (Open[i] execute, H/L[i] SL, Close[i] signal)
  6. Calcul metriques (Sharpe, Calmar, Sortino, PF, WR, DD, etc.)
  7. Rapport d'audit (divergences, phantom trades, confiance)

Tous les fixes D1-D13 sont integres dans ce pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path

from pineguard.config import PineGuardConfig, validate_config, DEFAULT_CONFIG
from pineguard.indicators.pmax import compute_pmax
from pineguard.indicators.atr import compute_atr
from pineguard.indicators.rsi import compute_rsi
from pineguard.indicators.bollinger import compute_bollinger
from pineguard.indicators.donchian import compute_donchian
from pineguard.indicators.adx import compute_adx
from pineguard.engine.backtest_engine import BacktestEngine
from pineguard.metrics.performance import compute_all_metrics
from pineguard.filters.entry_allowed import compute_entry_allowed
from pineguard.utils.data_loader import generate_sample_data, load_csv


def _compute_entry_signals(
    df: pd.DataFrame,
    config: PineGuardConfig,
    pmax_result: Dict,
) -> Dict[str, pd.Series]:
    """
    Compute entry signals based on configured entry type.

    Returns dict with 'entry_long' and 'entry_short' boolean Series.
    """
    entry_type = config.entry_type

    if entry_type == 'Donc+Chikou':
        from pineguard.entries.donchian_chikou import compute_donchian_chikou_signals
        signals = compute_donchian_chikou_signals(
            df['High'], df['Low'], df['Close'],
            don_length=config.don_length,
            chikou_shift=config.chikou_shift,
        )
        return {
            'entry_long': signals['don_long'],
            'entry_short': signals['don_short'],
            'details': signals,
        }

    elif entry_type == 'RSI Divergence':
        from pineguard.entries.rsi_divergence import compute_rsi_divergence_signals
        signals = compute_rsi_divergence_signals(
            df['High'], df['Low'], df['Close'],
            rsi_length=config.rsi_div_length,
            lookback=config.rsi_div_lookback,
            max_pivot_distance=config.rsi_div_max_distance,
        )
        return {
            'entry_long': signals['rsi_div_long'],
            'entry_short': signals['rsi_div_short'],
            'details': signals,
        }

    elif entry_type == 'Boll+SMA':
        from pineguard.entries.boll_sma import compute_boll_sma_signals
        signals = compute_boll_sma_signals(
            df['Close'],
            boll_length=config.boll_length,
            boll_mult=config.boll_mult,
            sma_length=config.boll_sma_length,
        )
        return {
            'entry_long': signals['boll_long'],
            'entry_short': signals['boll_short'],
            'details': signals,
        }

    elif entry_type == 'Fractal':
        from pineguard.entries.fractal import compute_fractal_signals
        signals = compute_fractal_signals(
            df['High'], df['Low'], df['Close'],
            patterns=config.fractal_patterns,
        )
        return {
            'entry_long': signals['fractal_long'],
            'entry_short': signals['fractal_short'],
            'details': signals,
        }

    elif entry_type == 'VWAP':
        from pineguard.entries.vwap import compute_vwap_signals
        signals = compute_vwap_signals(
            df['High'], df['Low'], df['Close'],
            volume=df.get('Volume', pd.Series(1, index=df.index)),
            session_start_hour=config.vwap_session_start,
        )
        return {
            'entry_long': signals['vwap_long'],
            'entry_short': signals['vwap_short'],
            'details': signals,
        }

    elif entry_type == 'ORB':
        from pineguard.entries.orb import compute_orb_signals
        signals = compute_orb_signals(
            df['High'], df['Low'], df['Close'], df['Open'],
            orb_minutes=config.orb_minutes,
            session_start_hour=config.orb_session_hour,
            session_start_minute=config.orb_session_minute,
            reproduce_pine_bug=False,  # Use CORRECTED logic
        )
        return {
            'entry_long': signals['orb_long'],
            'entry_short': signals['orb_short'],
            'details': signals,
        }

    elif entry_type == 'MM Cross':
        from pineguard.entries.mm_cross import compute_mm_cross_signals
        signals = compute_mm_cross_signals(
            df['Close'],
            ma1_type=config.mm_cross_ma1_type,
            ma1_length=config.mm_cross_ma1_len,
            ma2_type=config.mm_cross_ma2_type,
            ma2_length=config.mm_cross_ma2_len,
        )
        return {
            'entry_long': signals['mm_cross_long'],
            'entry_short': signals['mm_cross_short'],
            'details': signals,
        }

    else:
        raise ValueError(f"Unknown entry type: '{entry_type}'. "
                         f"Valid: Donc+Chikou, RSI Divergence, Boll+SMA, Fractal, VWAP, ORB, MM Cross")


def _compute_filters(
    df: pd.DataFrame,
    config: PineGuardConfig,
    pmax_result: Dict,
) -> Dict[str, pd.Series]:
    """
    Compute all filters and entry_allowed gate.

    Integrates:
      - PMax direction (component 1)
      - LT trend filter (component 3) — if enabled
      - MT trend filter (component 4) — if enabled
      - ADX regime filter — if enabled
    """
    lt_result = None
    mt_result = None

    if config.lt_filter_enabled:
        from pineguard.filters.trend_filter import compute_lt_filter
        lt_result = compute_lt_filter(
            df, lt_tf=config.lt_filter_tf,
            pmax_length=config.lt_pmax_length,
            pmax_mult=config.lt_pmax_mult,
            pmax_ma_type=config.lt_pmax_ma_type,
        )

    if config.mt_filter_enabled:
        from pineguard.filters.trend_filter import compute_mt_filter
        mt_result = compute_mt_filter(
            df, mt_tf=config.mt_filter_tf,
            pmax_length=config.mt_pmax_length,
            pmax_mult=config.mt_pmax_mult,
            pmax_ma_type=config.mt_pmax_ma_type,
        )

    # ADX regime filter
    adx_regime = None
    if config.adx_filter_enabled:
        from pineguard.filters.adx_regime import compute_adx_regime_filter
        adx_regime = compute_adx_regime_filter(
            df['High'], df['Low'], df['Close'],
            adx_length=config.adx_length,
            adx_smoothing=config.adx_smoothing,
            threshold=config.adx_threshold,
            entry_type=config.entry_type,
        )

    # Compute entry_allowed gate (fix B9)
    gate = compute_entry_allowed(
        pmax_dir=pmax_result['dir'],
        lt_filter_long=lt_result['lt_long'] if lt_result else None,
        lt_filter_short=lt_result['lt_short'] if lt_result else None,
        mt_filter_long=mt_result['mt_long'] if mt_result else None,
        mt_filter_short=mt_result['mt_short'] if mt_result else None,
        lt_enabled=config.lt_filter_enabled,
        mt_enabled=config.mt_filter_enabled,
    )

    # Combine ADX regime with entry_allowed
    ea_long = gate['entry_allowed_long']
    ea_short = gate['entry_allowed_short']
    if adx_regime is not None:
        ea_long = ea_long & adx_regime['regime_allowed']
        ea_short = ea_short & adx_regime['regime_allowed']

    return {
        'entry_allowed_long': ea_long,
        'entry_allowed_short': ea_short,
        'lt_filter': lt_result,
        'mt_filter': mt_result,
        'adx_regime': adx_regime,
        'blocked_by': gate['blocked_by'],
    }


def run_backtest(
    df: Optional[pd.DataFrame] = None,
    config: Optional[PineGuardConfig] = None,
    csv_path: Optional[str] = None,
    n_sample_bars: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute un backtest complet avec le pipeline PF AI Lab corrige.

    Pipeline:
      Data -> PMax -> Entry Signals -> Filters -> entry_allowed -> Backtest -> Metrics

    Args:
        df: DataFrame OHLCV (si None, genere des donnees sample ou charge csv_path)
        config: PineGuardConfig (si None, utilise DEFAULT_CONFIG)
        csv_path: Chemin vers fichier CSV OHLCV
        n_sample_bars: Nombre de barres sample si pas de data (default 1000)
        verbose: Afficher les messages de progression

    Returns:
        Dict complet avec:
            'trades': Liste des trades
            'equity': Courbe d'equity
            'metrics': Metriques completes
            'signals': Signaux d'entree bruts
            'filters': Resultats des filtres
            'pmax': Resultat PMax
            'config': Configuration utilisee
            'validation': Resultat de validation
    """
    # Config
    if config is None:
        config = DEFAULT_CONFIG

    validation = validate_config(config)
    if not validation['valid']:
        if verbose:
            print(f"  CONFIG INVALIDE: {validation['errors']}")
        return {'error': 'invalid_config', 'validation': validation}
    if verbose:
        print(f"PF AI Lab {__import__('pf_ai_lab').__version__} | "
              f"{config.instrument} {config.timeframe} | {config.entry_type}")
        for w in validation['warnings']:
            print(f"  {w}")

    # Data
    if df is None:
        if csv_path:
            df = load_csv(csv_path)
            if verbose:
                print(f"  Donnees chargees: {len(df)} barres depuis {csv_path}")
        else:
            df = generate_sample_data(
                n_bars=n_sample_bars, freq='1h',
                base_price=5000, volatility=0.001, seed=42,
            )
            if verbose:
                print(f"  Donnees sample generees: {n_sample_bars} barres H1")

    # PMax
    pmax_result = compute_pmax(
        df['High'], df['Low'], df['Close'],
        length=config.pmax_length,
        multiplier=config.pmax_multiplier,
        ma_type=config.pmax_ma_type,
        volume=df.get('Volume'),
    )
    if verbose:
        print(f"  PMax: {int(pmax_result['buy_trigger'].sum())} buy / "
              f"{int(pmax_result['sell_trigger'].sum())} sell triggers")

    # ATR (pour SL)
    atr = compute_atr(df['High'], df['Low'], df['Close'], 14)

    # Entry signals
    entry_result = _compute_entry_signals(df, config, pmax_result)
    raw_long = entry_result['entry_long']
    raw_short = entry_result['entry_short']

    # Combine with PMax direction (fundamental filter)
    combined_long = raw_long & (pmax_result['dir'] == 1)
    combined_short = raw_short & (pmax_result['dir'] == -1)

    if verbose:
        print(f"  Signaux bruts: {int(raw_long.sum())}L / {int(raw_short.sum())}S")
        print(f"  Apres PMax:    {int(combined_long.sum())}L / {int(combined_short.sum())}S")

    # Filters + entry_allowed gate (fix B9)
    filter_result = _compute_filters(df, config, pmax_result)
    ea_long = filter_result['entry_allowed_long']
    ea_short = filter_result['entry_allowed_short']

    if verbose:
        blocked = filter_result['blocked_by']
        print(f"  Filtres actifs: LT={config.lt_filter_enabled} "
              f"MT={config.mt_filter_enabled} ADX={config.adx_filter_enabled}")

    # Backtest 3 phases
    engine = BacktestEngine(
        instrument=config.instrument,
        sl_mode=config.sl_mode,
        sl_pct=config.sl_pct,
        initial_capital=config.initial_capital,
        position_size_pct=config.position_size_pct,
        enable_reversal=config.enable_reversal,
    )

    bt_result = engine.run(
        df=df,
        entry_long=combined_long,
        entry_short=combined_short,
        entry_allowed_long=ea_long,
        entry_allowed_short=ea_short,
        atr=atr,
    )

    # Metrics
    metrics = compute_all_metrics(
        bt_result['equity'], bt_result['trades'],
        periods_per_year=_periods_per_year(config.timeframe),
    )

    if verbose:
        print(f"\n  === RESULTATS ===")
        print(f"  Trades: {metrics['total_trades']} | "
              f"WR: {metrics['win_rate']}% | "
              f"PF: {metrics['profit_factor']} | "
              f"Sharpe: {metrics['sharpe_ratio']}")
        print(f"  PnL: {metrics['total_pnl']} | "
              f"Max DD: {metrics['max_drawdown_pct']}% | "
              f"Calmar: {metrics['calmar_ratio']}")

    return {
        'trades': bt_result['trades'],
        'equity': bt_result['equity'],
        'trade_count': bt_result['trade_count'],
        'metrics': metrics,
        'signals': entry_result,
        'filters': filter_result,
        'pmax': pmax_result,
        'atr': atr,
        'config': config,
        'validation': validation,
        'data': df,
    }


def run_full_pipeline(
    df: Optional[pd.DataFrame] = None,
    config: Optional[PineGuardConfig] = None,
    csv_path: Optional[str] = None,
    run_audit: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline complet: Backtest + Audit PineGuard.

    Inclut:
      1. Backtest standard (run_backtest)
      2. Inventaire divergences D1-D13
      3. Rapport de confiance par entry_type
      4. Estimation phantom trades
    """
    result = run_backtest(df=df, config=config, csv_path=csv_path, verbose=verbose)

    if 'error' in result:
        return result

    if run_audit:
        from pineguard.audit.divergence_tracker import DivergenceTracker
        from pineguard.audit.report_generator import AuditReportGenerator
        from pineguard import CONFIDENCE_MATRIX

        tracker = DivergenceTracker()
        summary = tracker.summary()

        conf = CONFIDENCE_MATRIX.get(result['config'].entry_type, {})
        confidence = conf.get('confidence', None)

        report_gen = AuditReportGenerator(
            instrument=result['config'].instrument,
            period=f"{len(result['data'])} bars {result['config'].timeframe}",
            entry_type=result['config'].entry_type,
            config=result['config'].to_dict(),
        )
        report_gen.add_divergence_inventory(tracker)
        audit_report = report_gen.generate()

        result['audit'] = {
            'divergence_summary': summary,
            'confidence': confidence,
            'confidence_label': conf.get('label', '?'),
            'report': audit_report,
            'phantom_trades_estimate': summary['estimated_total_phantom_trades'],
        }

        if verbose:
            print(f"\n  === AUDIT PINEGUARD ===")
            print(f"  Confiance {result['config'].entry_type}: {conf.get('label', '?')}")
            print(f"  Divergences: {summary['total']} "
                  f"({summary['by_severity'].get('BLOQUANT', 0)} BLOQ, "
                  f"{summary['by_severity'].get('HAUTE', 0)} HAUTE)")
            print(f"  Phantom trades estimes: ~{summary['estimated_total_phantom_trades']}")

    return result


def _periods_per_year(timeframe: str) -> float:
    """Convert timeframe to approximate periods per year."""
    tf_map = {
        '1M': 525600,   # ~525K minute bars/year
        '5M': 105120,
        '15M': 35040,
        '30M': 17520,
        '1H': 6240,     # ~260 days × 24h (or 252 × ~24h)
        '2H': 3120,
        '4H': 1560,
        '1D': 252,
        'D': 252,
        '1W': 52,
        'W': 52,
    }
    return tf_map.get(timeframe, 252)
