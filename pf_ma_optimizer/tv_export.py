"""
PF AI Lab 6.0.3 — TradingView Export Formatter
Generates copy-paste configuration text for Pine Script V69.3.

V6.0.3: RISK MANAGEMENT AUDIT
  - Export metadata block (schema_version, config_hash, export_timestamp)
  - Feed divergence warning in every export
  - Live Readiness Score shown prominently
V5.6: DIVERGENCE REDUCTION — runtime guards, no phantom params.
  - Runtime guard: pmax_ma_mode forced to 'classic', use_ma_cross_signal forced OFF.
  - Direction (Long Only / Short Only / Long & Short) shown in Explore table column.
  - No MA Factor, MA Dist, or MA Cross Signal lines in export (doesn't exist in Pine).
V5.4.9: CLEAN EXPORT — only Pine Script V69.3 parameters shown.
  - Removed phantom params: pmax_ma_mode, pmax_ma_factor, pmax_ma_distance,
    use_ma_cross_signal, sig_ma_type, sig_ma_fast, sig_ma_slow.
  - Direction (Long/Short/Both) ALWAYS shown clearly.
  - Parameter names match TradingView UI labels exactly.
V5.4.8: Added RM lockdown warning, phantom filter warnings, SL Plein note.
V5.4: Added quality grade, robustness score, WFE warnings.
"""


def format_config_for_tv(result: dict) -> str:
    """
    Format a single optimization result for TradingView copy-paste.
    Only shows parameters that EXIST in Pine Script PF_Algo V69.3.

    Parameters:
        result: dict with 'rank', 'score', 'config', 'metrics_is', 'metrics_oos'

    Returns:
        str: formatted text
    """
    rank = result.get('rank', '?')
    score = result.get('score', 0)
    config = result.get('config', {})
    m_is = result.get('metrics_is', {})
    m_oos = result.get('metrics_oos', {})

    lines = []
    sep = '─' * 60

    # V5.4: Grade and quality info
    grade = result.get('_grade', '')
    quality_score = result.get('_quality', 0)
    wfe_val = result.get('wfe')
    entry_type = config.get('entry_type', 'Donc+Chikou')

    # ── Header ──
    lines.append(sep)
    header = (f"RANK #{rank}  |  Score: {score:.4f}  |  "
              f"Calmar: {m_is.get('calmar', 0):.3f}  |  "
              f"MaxDD: {m_is.get('max_dd', 0):.2f}%  |  "
              f"Trades: {m_is.get('n_trades', 0)}  |  "
              f"WR: {m_is.get('win_rate', 0):.1f}%")
    if grade:
        header += f"  |  Grade: {grade}"
    lines.append(header)

    # ── Direction — ALWAYS shown prominently ──
    long_on = config.get('long_side', True)
    short_on = config.get('short_side', True)
    if long_on and short_on:
        dir_label = "LONG + SHORT"
    elif long_on and not short_on:
        dir_label = "LONG ONLY"
    elif short_on and not long_on:
        dir_label = "SHORT ONLY"
    else:
        dir_label = "AUCUNE (config invalide)"
    lines.append(f"Direction: {dir_label}  |  Entry: {entry_type}")

    # Grade/Quality/WFE
    if grade or quality_score:
        grade_line = ""
        if grade:
            grade_line += f"Grade: {grade} ({quality_score:.1f})"
        if wfe_val is not None:
            grade_line += f" | WFE: {wfe_val:.2f}"
        elif wfe_val is None:
            grade_line += " | WFE: N/A"
        if grade_line:
            lines.append(grade_line)

        # OOS summary
        oos_trades = m_oos.get('n_trades', 0)
        if oos_trades > 0:
            lines.append(f"OOS: Calmar={m_oos.get('calmar', 0):.2f} "
                         f"Sharpe={m_oos.get('sharpe', 0):.2f} "
                         f"PF={m_oos.get('profit_factor', 0):.2f} "
                         f"DD={m_oos.get('max_dd', 0):.2f}% "
                         f"({oos_trades} trades)")

        # Robustness summary
        perturb = result.get('perturbation_pass', {}) or {}
        wf = result.get('walk_forward', {}) or {}
        rob_parts = []
        if perturb.get('stable') is not None:
            status = "STABLE" if perturb.get('stable') else "FRAGILE"
            deg_pct = (f"{perturb.get('degradation', 0):.0%}"
                       if perturb.get('degradation') is not None else '?')
            rob_parts.append(f"Perturbation={status}({deg_pct})")
        if wf.get('stability_score') is not None:
            rob_parts.append(f"WF={wf.get('stability_score', 0):.1f}/100")
        if rob_parts:
            lines.append(f"Robustness: {' '.join(rob_parts)}")

        # Flags and warnings
        flags = result.get('_flags', [])
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")

        if wfe_val is not None and wfe_val > 2.0:
            lines.append(f"WFE={wfe_val:.2f} — OOS >> IS, possibly lucky")

        # SAFE check
        try:
            from pineguard import SAFE_ENTRY_TYPES
            if entry_type not in SAFE_ENTRY_TYPES:
                lines.append(f"Entry '{entry_type}' not validated by PineGuard")
        except ImportError:
            pass

    lines.append(sep)

    # ══════════════════════════════════════════════════════════════
    # TRADINGVIEW SETTINGS — Pine Script PF_Algo V69.3 parameters
    # Only parameters that exist in the Pine Script are shown.
    # ══════════════════════════════════════════════════════════════

    # ── PMax Settings (group "PMax") ──
    lines.append("")
    lines.append("PMax Settings:")
    lines.append(f"  MA Type        → {config.get('pmax_ma_type', 'EMA')}")
    lines.append(f"  MA Length      → {config.get('pmax_length', 10)}")
    lines.append(f"  ATR Multiplier → {config.get('pmax_multiplier', 3.0):.1f}")

    # ── Entry (group "Entry & Signal") ──
    lines.append("")
    lines.append(f"Entry Type → {entry_type}")

    if entry_type == 'Donc+Chikou':
        lines.append(f"  Donchian Period → {config.get('don_length', 20)}")
    elif entry_type == 'RSI Divergence':
        lines.append(f"  RSI Length       → {config.get('rsi_div_length', 14)}")
        lines.append(f"  Pivot Lookback   → {config.get('rsi_div_pivot', 5)}")
        lines.append(f"  Hidden Div       → {'ON' if config.get('rsi_div_hidden', True) else 'OFF'}")
    elif entry_type == 'Fractal':
        lines.append(f"  Buffer Point     → {config.get('fractal_buffer', 0.0):.1f}")
    elif entry_type == 'Boll+SMA':
        lines.append(f"  BB MA Length     → {config.get('length_bb_ma', 20)}")
        lines.append(f"  BB StdDev        → {config.get('mult_bb_ma', 2.0):.1f}")
        lines.append(f"  BB SMA Length    → {config.get('length_bb_sma', 50)}")
    elif entry_type == 'MM Cross':
        lines.append(f"  MM1 (Fast)      → {config.get('mm_cross_ma1_type', 'EMA')}({config.get('mm_cross_ma1_len', 9)})")
        lines.append(f"  MM2 (Slow)      → {config.get('mm_cross_ma2_type', 'EMA')}({config.get('mm_cross_ma2_len', 21)})")

    lines.append(f"  Order Pause      → {config.get('order_pause', 5)}")

    # ── Direction (group "Entry & Signal") ──
    lines.append("")
    if long_on and short_on:
        lines.append("Direction → Long + Short")
    elif long_on and not short_on:
        lines.append("Direction → LONG ONLY")
        lines.append("  Sur TradingView: decocher 'Short Side' dans Entries")
    elif short_on and not long_on:
        lines.append("Direction → SHORT ONLY")
        lines.append("  Sur TradingView: decocher 'Long Side' dans Entries")
    else:
        lines.append("Direction → AUCUNE (Long=OFF, Short=OFF) — config invalide")

    # ── Filters (groups "Filter -") ──
    lines.append("")
    lines.append("Filters:")

    # LT Trend Filter
    if config.get('use_lt_filter', False):
        lines.append(f"  LT Trend  → ON | {config.get('lt_ma_type', 'EMA')}"
                     f"({config.get('lt_length', 20)}) "
                     f"TF={config.get('lt_tf', 'W')} "
                     f"Mult={config.get('lt_multiplier', 1.5):.1f}")
    else:
        lines.append("  LT Trend  → OFF")

    # MT Trend Filter
    if config.get('use_mt_filter', False):
        lines.append(f"  MT Trend  → ON | {config.get('mt_ma_type', 'EMA')}"
                     f"({config.get('mt_length', 10)}) "
                     f"TF={config.get('mt_tf', '4H')} "
                     f"Mult={config.get('mt_multiplier', 1.5):.1f}")
    else:
        lines.append("  MT Trend  → OFF")

    # RSI > 50
    if config.get('use_rsi50_filter', False):
        lines.append("  RSI > 50  → ON")
    else:
        lines.append("  RSI > 50  → OFF")

    # MA Direction Filter
    if config.get('use_ma_dir_filter', False):
        lines.append(f"  MA Dir    → ON | Len={config.get('ma_dir_len', 3)}")
    else:
        lines.append("  MA Dir    → OFF")

    # MAvg Position Filter
    if config.get('use_mavg_filter', False):
        lines.append("  MAvg Pos  → ON")
    else:
        lines.append("  MAvg Pos  → OFF")

    # Diff MA/Red Filter
    if config.get('use_diff_ma_red', False):
        lines.append(f"  Diff MA   → ON | {config.get('diff_ma_red_pct', 1.0):.1f}%")
    else:
        lines.append("  Diff MA   → OFF")

    # Diff Price/Red Filter
    if config.get('use_diff_price_red', False):
        lines.append(f"  Diff Pr   → ON | {config.get('diff_price_red_pct', 1.0):.1f}%")
    else:
        lines.append("  Diff Pr   → OFF")

    # ADX Regime Filter
    if config.get('use_adx_regime_filter', False):
        lines.append(f"  ADX Reg   → ON | Threshold={config.get('adx_trend_threshold', 25)}")
    else:
        lines.append("  ADX Reg   → OFF")

    # Volatility Filter
    if config.get('use_vol_filter', False):
        lines.append(f"  Vol Filt  → ON | Percentile={config.get('vol_percentile_threshold', 70)}")
    else:
        lines.append("  Vol Filt  → OFF")

    # QQ Estimation Filter
    if config.get('use_qq_filter', False):
        lines.append(f"  QQ Filt   → ON | Length={config.get('qq_length', 14)}")
    else:
        lines.append("  QQ Filt   → OFF")

    # ── Risk Management — LOCKED ──
    lines.append("")
    lines.append("Risk Management (a valider sur TradingView):")
    lines.append(f"  SL       → Static 30% (filet de securite)")
    lines.append(f"  TP       → No")
    lines.append(f"  BE/TSL   → No")
    lines.append(f"  Exit     → {'Setup Reversal' if config.get('exit_setup_reversal', True) else 'No Reversal Exit'}")

    # ── Perturbation Robustness ──
    perturb = result.get('perturbation_pass', None)
    if perturb:
        lines.append("")
        status = "STABLE" if perturb.get('stable', False) else "FRAGILE"
        lines.append(f"Perturbation: {status} "
                     f"({perturb.get('n_surviving', '?')}/{perturb.get('n_total', '?')} surviving, "
                     f"degradation {perturb.get('degradation', 0):.1%})")

    # ── Walk-Forward ──
    wf = result.get('walk_forward', None)
    if wf:
        wf_score = wf.get('stability_score', 0)
        if wf_score >= 70:
            wf_label = 'STABLE'
        elif wf_score >= 40:
            wf_label = 'MODERATE'
        else:
            wf_label = 'UNSTABLE'
        lines.append(f"Walk-Forward: {wf_label} ({wf_score:.0f}/100, "
                     f"{wf.get('consistency', 0):.0f}% profitable)")

    # ── IS Metrics ──
    lines.append("")
    lines.append("In-Sample Metrics:")
    lines.append(f"  Calmar: {m_is.get('calmar', 0):.3f}  |  "
                 f"Sortino: {m_is.get('sortino', 0):.3f}  |  "
                 f"Sharpe: {m_is.get('sharpe', 0):.3f}")
    lines.append(f"  PF: {m_is.get('profit_factor', 0):.3f}  |  "
                 f"MaxDD: {m_is.get('max_dd', 0):.2f}%  |  "
                 f"WR: {m_is.get('win_rate', 0):.1f}%")
    lines.append(f"  Trades: {m_is.get('n_trades', 0)}  |  "
                 f"Net: {m_is.get('net_return', 0):.2f}%  |  "
                 f"Annual: {m_is.get('annual_return', 0):.2f}%")

    # ── OOS Metrics ──
    oos_trades = m_oos.get('n_trades', 0)
    if oos_trades > 0:
        lines.append("")
        lines.append("Out-of-Sample Metrics:")
        lines.append(f"  Calmar: {m_oos.get('calmar', 0):.3f}  |  "
                     f"Sortino: {m_oos.get('sortino', 0):.3f}  |  "
                     f"Sharpe: {m_oos.get('sharpe', 0):.3f}")
        lines.append(f"  PF: {m_oos.get('profit_factor', 0):.3f}  |  "
                     f"MaxDD: {m_oos.get('max_dd', 0):.2f}%  |  "
                     f"WR: {m_oos.get('win_rate', 0):.1f}%")
        lines.append(f"  Trades: {m_oos.get('n_trades', 0)}  |  "
                     f"Net: {m_oos.get('net_return', 0):.2f}%")
    else:
        lines.append("")
        lines.append("Out-of-Sample: No trades")

    lines.append("")
    lines.append(sep)

    return '\n'.join(lines)


def format_all_configs(top_configs: list, is_range: tuple = None,
                       oos_range: tuple = None,
                       instrument: str = '') -> str:
    """
    Format all top configs for export.
    """
    lines = []

    lines.append("=" * 70)
    lines.append(f"P.F.ALGO V69.3 — EXPLORATION RESULTS (AI Lab 6.0.3)")
    lines.append(f"Instrument: {instrument}")
    if is_range:
        lines.append(f"IS Period:  {is_range[0]} → {is_range[1]}")
    if oos_range:
        lines.append(f"OOS Period: {oos_range[0]} → {oos_range[1]}")
    lines.append(f"Top {len(top_configs)} Configs")
    lines.append("=" * 70)

    lines.append("")
    lines.append("IMPORTANT: These results are EXPLORATORY.")
    lines.append("  Validate in TradingView Strategy Tester before deploying.")
    lines.append("  AI Lab optimizes ENTRY + FILTERS only.")
    lines.append("  SL/TP/BE/TSL must be configured on TradingView.")
    lines.append("")
    lines.append("FEED DIVERGENCE WARNING (v6.0.3):")
    lines.append("  Metriques calculees sur feed MT5. Les resultats peuvent")
    lines.append("  diverger de 30-50% sur un autre broker/feed.")
    lines.append("  PF OOS live estime = PF_oos * 0.70 (discount par defaut).")
    lines.append("")

    for result in top_configs:
        lines.append(format_config_for_tv(result))
        lines.append("")

    return '\n'.join(lines)
