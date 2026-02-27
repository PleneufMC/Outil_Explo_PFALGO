"""
PF AI Lab 5.4.8 — TradingView Export Formatter
Generates copy-paste configuration text for Pine Script V69.3.

V5.4.8: Added RM lockdown warning, phantom filter warnings, SL Plein philosophy note.
V5.4: Added quality grade, robustness score, WFE warnings, position sizing disclaimer.
"""


def format_config_for_tv(result: dict) -> str:
    """
    Format a single optimization result for TradingView copy-paste.

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

    # Header — V5.4 enriched
    lines.append(sep)
    header = (f"RANK #{rank}  |  Score: {score:.4f}  |  "
              f"Calmar: {m_is.get('calmar', 0):.3f}  |  "
              f"MaxDD: {m_is.get('max_dd', 0):.2f}%  |  "
              f"Trades: {m_is.get('n_trades', 0)}  |  "
              f"WR: {m_is.get('win_rate', 0):.1f}%")
    if grade:
        header += f"  |  Grade: {grade}"
    lines.append(header)

    # V5.4: Grade/Quality/WFE summary line
    if grade or quality_score:
        grade_line = f"Entry: {entry_type}"
        if grade:
            grade_line += f" | Grade: {grade} ({quality_score:.1f})"
        if wfe_val is not None:
            grade_line += f" | WFE: {wfe_val:.2f}"
        elif wfe_val is None:
            grade_line += " | WFE: N/A"
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
            deg_pct = f"{perturb.get('degradation', 0):.0%}" if perturb.get('degradation') is not None else '?'
            rob_parts.append(f"Perturbation={status}({deg_pct})")
        if wf.get('stability_score') is not None:
            rob_parts.append(f"WF={wf.get('stability_score', 0):.1f}/100")
        if rob_parts:
            lines.append(f"Robustness: {' '.join(rob_parts)}")

        # V5.4: Flags and warnings
        flags = result.get('_flags', [])
        if flags:
            lines.append(f"⚠ Flags: {', '.join(flags)}")

        # WFE warnings
        if wfe_val is not None and wfe_val > 2.0:
            lines.append(f"⚠ WFE={wfe_val:.2f} — OOS significantly better than IS, probably lucky")

        # SAFE check
        try:
            from pineguard import SAFE_ENTRY_TYPES
            if entry_type not in SAFE_ENTRY_TYPES:
                lines.append(f"⚠ Entry type '{entry_type}' not validated by PineGuard — Python results unreliable")
        except ImportError:
            pass

    lines.append(sep)

    # PMax Settings
    lines.append("")
    lines.append("📊 PMax Settings (Moving Average + ATR):")
    ma_mode = config.get('pmax_ma_mode', 'classic')
    lines.append(f"  MA Mode    → {ma_mode}")
    if ma_mode == 'classic':
        lines.append(f"  MA Type    → {config.get('pmax_ma_type', 'EMA')}")
        lines.append(f"  MA Length  → {config.get('pmax_length', 10)}")
    elif ma_mode in ('mult_int', 'mult_dec'):
        lines.append(f"  MA Factor  → {config.get('pmax_ma_factor', 1.0):.4f}")
    elif ma_mode == 'additive':
        lines.append(f"  MA Dist    → {config.get('pmax_ma_distance', 0.0):.1f}")
    lines.append(f"  Multiplier → {config.get('pmax_multiplier', 3.0):.1f}")

    # Entry
    lines.append("")
    entry_type = config.get('entry_type', 'Donc+Chikou')
    lines.append(f"🎯 Entry Type → {entry_type}")

    if entry_type == 'Donc+Chikou':
        lines.append(f"  Donchian Period → {config.get('don_length', 20)}")
    elif entry_type == 'RSI Divergence':
        lines.append(f"  RSI Length      → {config.get('rsi_div_length', 14)}")
        lines.append(f"  Pivot Lookback  → {config.get('rsi_div_pivot', 5)}")
        lines.append(f"  Hidden Div      → {'ON' if config.get('rsi_div_hidden', True) else 'OFF'}")
    elif entry_type == 'Fractal':
        lines.append(f"  Buffer Point    → {config.get('fractal_buffer', 0.0):.1f}")
    elif entry_type == 'Boll+SMA':
        lines.append(f"  BB MA Length    → {config.get('length_bb_ma', 20)}")
        lines.append(f"  BB StdDev       → {config.get('mult_bb_ma', 2.0):.1f}")
        lines.append(f"  BB SMA Length   → {config.get('length_bb_sma', 50)}")
    elif entry_type == 'MM Cross':
        lines.append(f"  MM1 (Fast)     → {config.get('mm_cross_ma1_type', 'EMA')}({config.get('mm_cross_ma1_len', 9)})")
        lines.append(f"  MM2 (Slow)     → {config.get('mm_cross_ma2_type', 'EMA')}({config.get('mm_cross_ma2_len', 21)})")

    lines.append(f"  Order Pause     → {config.get('order_pause', 5)}")

    # Independent MA Cross Signal
    if config.get('use_ma_cross_signal', False):
        lines.append("")
        lines.append("📐 MA Cross Signal (AND with PMax):")
        lines.append(f"  MA Type    → {config.get('sig_ma_type', 'EMA')}")
        lines.append(f"  Fast MA    → {config.get('sig_ma_fast', 9)}")
        lines.append(f"  Slow MA    → {config.get('sig_ma_slow', 21)}")
    else:
        lines.append("")
        lines.append("📐 MA Cross Signal → ❌ OFF")

    # Filters
    lines.append("")
    lines.append("🔍 Filters:")

    # LT Trend
    if config.get('use_lt_filter', False):
        lines.append(f"  LT Trend → ✅ ON | {config.get('lt_ma_type', 'EMA')}"
                     f"({config.get('lt_length', 20)}) "
                     f"{config.get('lt_tf', 'W')} "
                     f"×{config.get('lt_multiplier', 1.5):.1f}")
    else:
        lines.append("  LT Trend → ❌ OFF")

    # MT Trend
    if config.get('use_mt_filter', False):
        lines.append(f"  MT Trend → ✅ ON | {config.get('mt_ma_type', 'EMA')}"
                     f"({config.get('mt_length', 10)}) "
                     f"{config.get('mt_tf', '4H')} "
                     f"×{config.get('mt_multiplier', 1.5):.1f}")
    else:
        lines.append("  MT Trend → ❌ OFF")

    # RSI > 50
    if config.get('use_rsi50_filter', False):
        lines.append("  RSI>50   → ✅ ON")
    else:
        lines.append("  RSI>50   → ❌ OFF")

    # Diff MA/Red
    if config.get('use_diff_ma_red', False):
        lines.append(f"  Diff MA  → ✅ ON | {config.get('diff_ma_red_pct', 1.0):.1f}%")
    else:
        lines.append("  Diff MA  → ❌ OFF")

    # MA Direction
    if config.get('use_ma_dir_filter', False):
        lines.append(f"  MA Dir   → ✅ ON | Len={config.get('ma_dir_len', 3)}")
    else:
        lines.append("  MA Dir   → ❌ OFF")

    # MAvg Position
    if config.get('use_mavg_filter', False):
        lines.append("  MAvg Pos → ✅ ON")
    else:
        lines.append("  MAvg Pos → ❌ OFF")

    # Diff Price/Red
    if config.get('use_diff_price_red', False):
        lines.append(f"  Diff Pr  → ✅ ON | {config.get('diff_price_red_pct', 1.0):.1f}%")
    else:
        lines.append("  Diff Pr  → ❌ OFF")

    # Risk Management — V5.4.8: LOCKED (user validates on TV)
    lines.append("")
    lines.append("🛡️ Risk Management (NOT optimized — validate on TradingView):")
    lines.append("  SL Mode  → Static-% 30.0 (filet de sécurité / \"SL Plein\")")
    lines.append("  TP Mode  → No (configure on TV)")
    lines.append("  BE/TSL   → No (configure on TV)")
    lines.append(f"  Exit     → {'Setup Reversal' if config.get('exit_setup_reversal', True) else 'No Reversal Exit'}")
    lines.append("  ⚠️  SL/TP/BE/TSL must be tested EXCLUSIVELY in TradingView.")

    # Perturbation Robustness
    perturb = result.get('perturbation_pass', None)
    if perturb:
        lines.append("")
        lines.append("🔬 Perturbation Robustness (±10%):")
        status = "STABLE" if perturb.get('stable', False) else "FRAGILE"
        lines.append(f"  Status      → {'✅' if perturb.get('stable') else '⚠️'} {status}")
        lines.append(f"  Surviving   → {perturb.get('n_surviving', '?')}/{perturb.get('n_total', '?')}")
        lines.append(f"  Degradation → {perturb.get('degradation', 0):.1%}")
        lines.append(f"  Base Calmar → {perturb.get('base_calmar', 0):.3f}  |  Avg Perturbed → {perturb.get('avg_calmar', 0):.3f}")

    # Walk-Forward Analysis
    wf = result.get('walk_forward', None)
    if wf:
        lines.append("")
        wf_score = wf.get('stability_score', 0)
        if wf_score >= 70:
            wf_label = '✅ STABLE'
        elif wf_score >= 40:
            wf_label = '⚠️ MODERATE'
        else:
            wf_label = '❌ UNSTABLE'
        lines.append(f"📉 Walk-Forward Analysis ({wf.get('n_windows', 0)} windows):")
        lines.append(f"  Stability   → {wf_label} ({wf_score:.0f}/100)")
        lines.append(f"  Consistency → {wf.get('consistency', 0):.0f}% windows profitable")
        lines.append(f"  Avg OOS Calmar → {wf.get('avg_calmar', 0):.3f}  |  Avg OOS PF → {wf.get('avg_pf', 0):.3f}")
        lines.append(f"  Degradation → {'⚠️ YES — performance declining' if wf.get('degradation') else '✅ NO'}")

    # IS Metrics
    lines.append("")
    lines.append("📈 In-Sample Metrics:")
    lines.append(f"  Calmar: {m_is.get('calmar', 0):.3f}  |  "
                 f"Sortino: {m_is.get('sortino', 0):.3f}  |  "
                 f"Sharpe: {m_is.get('sharpe', 0):.3f}")
    lines.append(f"  PF: {m_is.get('profit_factor', 0):.3f}  |  "
                 f"MaxDD: {m_is.get('max_dd', 0):.2f}%  |  "
                 f"WR: {m_is.get('win_rate', 0):.1f}%")
    lines.append(f"  Trades: {m_is.get('n_trades', 0)}  |  "
                 f"Net: {m_is.get('net_return', 0):.2f}%  |  "
                 f"Annual: {m_is.get('annual_return', 0):.2f}%")

    # OOS Metrics
    lines.append("")
    oos_trades = m_oos.get('n_trades', 0)
    if oos_trades > 0:
        lines.append("📊 Out-of-Sample Metrics:")
        lines.append(f"  Calmar: {m_oos.get('calmar', 0):.3f}  |  "
                     f"Sortino: {m_oos.get('sortino', 0):.3f}  |  "
                     f"Sharpe: {m_oos.get('sharpe', 0):.3f}")
        lines.append(f"  PF: {m_oos.get('profit_factor', 0):.3f}  |  "
                     f"MaxDD: {m_oos.get('max_dd', 0):.2f}%  |  "
                     f"WR: {m_oos.get('win_rate', 0):.1f}%")
        lines.append(f"  Trades: {m_oos.get('n_trades', 0)}  |  "
                     f"Net: {m_oos.get('net_return', 0):.2f}%")
    else:
        lines.append("📊 Out-of-Sample: No trades")

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
    lines.append(f"P.F.ALGO V69.1 — EXPLORATION RESULTS")
    lines.append(f"Instrument: {instrument}")
    if is_range:
        lines.append(f"IS Period:  {is_range[0]} → {is_range[1]}")
    if oos_range:
        lines.append(f"OOS Period: {oos_range[0]} → {oos_range[1]}")
    lines.append(f"Top {len(top_configs)} Configs")
    lines.append("=" * 70)

    lines.append("")
    lines.append("⚠️  IMPORTANT: These results are EXPLORATORY.")
    lines.append("    Python results may differ ±10-30% from TradingView.")
    lines.append("    ALWAYS validate in TradingView Strategy Tester before deploying.")
    lines.append("")
    lines.append("    POSITION SIZING: These results assume 1 contract / 1 lot.")
    lines.append("    Real position sizing must account for account size, max risk per trade")
    lines.append("    (typically 1-2% of equity), and instrument-specific contract values.")
    lines.append("    Do NOT trade the full account on a single strategy or instrument.")
    lines.append("")
    lines.append("    RISK MANAGEMENT: AI Lab uses SL=30% (safety net) and NO Take Profit.")
    lines.append("    The optimizer finds ENTRY signals + FILTERS only. SL/TP/BE/TSL must be")
    lines.append("    configured and validated EXCLUSIVELY in TradingView Strategy Tester.")
    lines.append("    Philosophy: exits via setup reversal or filter change, NOT tight stops.")
    lines.append("")
    lines.append("    FILTERS NOT SIMULATED: VIX, Macro, Correlation, MTF Momentum,")
    lines.append("    Gap Protection, Session timing are NOT computed by AI Lab.")
    lines.append("    If you enable these on TV, performance WILL differ from AI Lab results.")
    lines.append("")

    for result in top_configs:
        lines.append(format_config_for_tv(result))
        lines.append("")

    return '\n'.join(lines)
