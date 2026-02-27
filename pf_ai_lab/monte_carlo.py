"""
PF AI Lab 5.0.4 — Simulation Monte Carlo.

Methodes:
  1. Trade Shuffling: permute l'ordre des trades pour estimer la distribution
     du max drawdown, du PnL final, et des metriques de risque.
  2. Returns Bootstrap: resample avec remplacement les rendements.
  3. Noise Injection: ajoute du bruit aux prix pour tester la robustesse.

Output:
  - Distribution du max drawdown (percentiles 5/25/50/75/95)
  - Probabilite de ruine (equity < seuil)
  - Intervalles de confiance pour Sharpe, PF, WR
  - Histogramme des PnL finaux
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    n_simulations: int
    method: str

    # PnL distribution
    pnl_mean: float
    pnl_std: float
    pnl_percentiles: Dict[str, float]  # {5: ..., 25: ..., 50: ..., 75: ..., 95: ...}

    # Drawdown distribution
    max_dd_mean: float
    max_dd_std: float
    max_dd_percentiles: Dict[str, float]

    # Risk metrics
    probability_of_ruin: float  # P(equity < ruin_threshold)
    var_95: float               # Value-at-Risk 95%
    cvar_95: float              # Conditional VaR 95%

    # Sharpe distribution
    sharpe_mean: float
    sharpe_ci_95: tuple         # (lower, upper)

    # Raw data for plotting
    equity_curves: Optional[np.ndarray] = None  # shape (n_sims, n_bars)
    pnl_distribution: Optional[np.ndarray] = None


def monte_carlo_trade_shuffle(
    trades: list,
    initial_capital: float = 100000.0,
    n_simulations: int = 1000,
    ruin_threshold: float = 0.5,
    seed: int = 42,
    store_curves: bool = False,
) -> MonteCarloResult:
    """
    Monte Carlo par permutation de l'ordre des trades.

    Prend les PnL reels des trades et permute leur ordre N fois
    pour estimer la distribution du drawdown et du PnL final.

    Args:
        trades: Liste de Trade objects (doit avoir .pnl_points)
        initial_capital: Capital initial
        n_simulations: Nombre de simulations (default 1000)
        ruin_threshold: Seuil de ruine en % du capital (default 0.5 = 50%)
        seed: Random seed
        store_curves: Stocker les courbes d'equity (attention memoire)

    Returns:
        MonteCarloResult
    """
    rng = np.random.RandomState(seed)

    pnl_list = np.array([t.pnl_points for t in trades])
    n_trades = len(pnl_list)

    if n_trades == 0:
        return _empty_mc_result(n_simulations, 'trade_shuffle')

    final_pnls = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)
    sharpes = np.zeros(n_simulations)
    curves = np.zeros((n_simulations, n_trades + 1)) if store_curves else None

    ruin_level = initial_capital * ruin_threshold

    for sim in range(n_simulations):
        shuffled = rng.permutation(pnl_list)
        equity = np.empty(n_trades + 1)
        equity[0] = initial_capital
        equity[1:] = initial_capital + np.cumsum(shuffled)

        if store_curves:
            curves[sim] = equity

        final_pnls[sim] = equity[-1] - initial_capital

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dds[sim] = abs(dd.min()) * 100

        # Sharpe (simplified: mean/std of trade PnL)
        if shuffled.std() > 0:
            sharpes[sim] = shuffled.mean() / shuffled.std() * np.sqrt(252)
        else:
            sharpes[sim] = 0

    return MonteCarloResult(
        n_simulations=n_simulations,
        method='trade_shuffle',
        pnl_mean=round(float(final_pnls.mean()), 2),
        pnl_std=round(float(final_pnls.std()), 2),
        pnl_percentiles={
            '5': round(float(np.percentile(final_pnls, 5)), 2),
            '25': round(float(np.percentile(final_pnls, 25)), 2),
            '50': round(float(np.percentile(final_pnls, 50)), 2),
            '75': round(float(np.percentile(final_pnls, 75)), 2),
            '95': round(float(np.percentile(final_pnls, 95)), 2),
        },
        max_dd_mean=round(float(max_dds.mean()), 2),
        max_dd_std=round(float(max_dds.std()), 2),
        max_dd_percentiles={
            '5': round(float(np.percentile(max_dds, 5)), 2),
            '25': round(float(np.percentile(max_dds, 25)), 2),
            '50': round(float(np.percentile(max_dds, 50)), 2),
            '75': round(float(np.percentile(max_dds, 75)), 2),
            '95': round(float(np.percentile(max_dds, 95)), 2),
        },
        probability_of_ruin=round(float((final_pnls < -initial_capital * (1 - ruin_threshold)).mean()), 4),
        var_95=round(float(np.percentile(final_pnls, 5)), 2),  # 5th percentile = 95% VaR
        cvar_95=round(float(final_pnls[final_pnls <= np.percentile(final_pnls, 5)].mean()) if len(final_pnls[final_pnls <= np.percentile(final_pnls, 5)]) > 0 else 0, 2),
        sharpe_mean=round(float(sharpes.mean()), 3),
        sharpe_ci_95=(round(float(np.percentile(sharpes, 2.5)), 3),
                      round(float(np.percentile(sharpes, 97.5)), 3)),
        equity_curves=curves,
        pnl_distribution=final_pnls,
    )


def monte_carlo_noise_injection(
    df: pd.DataFrame,
    config: 'PineGuardConfig',
    noise_std_pct: float = 0.001,
    n_simulations: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> MonteCarloResult:
    """
    Monte Carlo par injection de bruit dans les prix.

    Ajoute un bruit gaussien aux OHLC pour tester la robustesse
    des signaux face aux micro-variations de donnees (simule D1 bid/mid).

    Args:
        df: DataFrame OHLCV original
        config: PineGuardConfig
        noise_std_pct: Ecart-type du bruit en % du prix (default 0.1%)
        n_simulations: Nombre de simulations
        seed: Random seed
        verbose: Afficher progression

    Returns:
        MonteCarloResult
    """
    from pf_ai_lab.runner import run_backtest

    rng = np.random.RandomState(seed)

    final_pnls = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)
    sharpes = np.zeros(n_simulations)
    trade_counts = np.zeros(n_simulations)

    for sim in range(n_simulations):
        # Add noise to OHLC
        df_noisy = df.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            noise = 1 + rng.normal(0, noise_std_pct, len(df))
            df_noisy[col] = df[col] * noise

        # Ensure OHLC consistency after noise
        df_noisy['High'] = df_noisy[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df_noisy['Low'] = df_noisy[['Open', 'High', 'Low', 'Close']].min(axis=1)

        result = run_backtest(df=df_noisy, config=config, verbose=False)
        if 'error' in result:
            continue

        metrics = result['metrics']
        final_pnls[sim] = metrics.get('total_pnl', 0)
        max_dds[sim] = metrics.get('max_drawdown_pct', 0)
        sharpes[sim] = metrics.get('sharpe_ratio', 0)
        trade_counts[sim] = metrics.get('total_trades', 0)

        if verbose and (sim + 1) % 10 == 0:
            print(f"  MC Noise: {sim+1}/{n_simulations}")

    return MonteCarloResult(
        n_simulations=n_simulations,
        method='noise_injection',
        pnl_mean=round(float(final_pnls.mean()), 2),
        pnl_std=round(float(final_pnls.std()), 2),
        pnl_percentiles={
            '5': round(float(np.percentile(final_pnls, 5)), 2),
            '25': round(float(np.percentile(final_pnls, 25)), 2),
            '50': round(float(np.percentile(final_pnls, 50)), 2),
            '75': round(float(np.percentile(final_pnls, 75)), 2),
            '95': round(float(np.percentile(final_pnls, 95)), 2),
        },
        max_dd_mean=round(float(max_dds.mean()), 2),
        max_dd_std=round(float(max_dds.std()), 2),
        max_dd_percentiles={
            '5': round(float(np.percentile(max_dds, 5)), 2),
            '25': round(float(np.percentile(max_dds, 25)), 2),
            '50': round(float(np.percentile(max_dds, 50)), 2),
            '75': round(float(np.percentile(max_dds, 75)), 2),
            '95': round(float(np.percentile(max_dds, 95)), 2),
        },
        probability_of_ruin=0.0,  # Not applicable for noise injection
        var_95=round(float(np.percentile(final_pnls, 5)), 2),
        cvar_95=round(float(final_pnls[final_pnls <= np.percentile(final_pnls, 5)].mean()) if len(final_pnls[final_pnls <= np.percentile(final_pnls, 5)]) > 0 else 0, 2),
        sharpe_mean=round(float(sharpes.mean()), 3),
        sharpe_ci_95=(round(float(np.percentile(sharpes, 2.5)), 3),
                      round(float(np.percentile(sharpes, 97.5)), 3)),
    )


def _empty_mc_result(n_sims: int, method: str) -> MonteCarloResult:
    """Empty MC result when no trades available."""
    empty_pct = {'5': 0, '25': 0, '50': 0, '75': 0, '95': 0}
    return MonteCarloResult(
        n_simulations=n_sims, method=method,
        pnl_mean=0, pnl_std=0, pnl_percentiles=empty_pct,
        max_dd_mean=0, max_dd_std=0, max_dd_percentiles=empty_pct,
        probability_of_ruin=0, var_95=0, cvar_95=0,
        sharpe_mean=0, sharpe_ci_95=(0, 0),
    )


def format_mc_report(result: MonteCarloResult) -> str:
    """Format Monte Carlo result as readable report."""
    lines = [
        f"Monte Carlo Simulation ({result.method})",
        f"  Simulations: {result.n_simulations}",
        f"",
        f"  PnL Distribution:",
        f"    Mean: {result.pnl_mean:,.2f} (std: {result.pnl_std:,.2f})",
        f"    P5/P25/P50/P75/P95: "
        f"{result.pnl_percentiles['5']:,.0f} / "
        f"{result.pnl_percentiles['25']:,.0f} / "
        f"{result.pnl_percentiles['50']:,.0f} / "
        f"{result.pnl_percentiles['75']:,.0f} / "
        f"{result.pnl_percentiles['95']:,.0f}",
        f"",
        f"  Max Drawdown Distribution:",
        f"    Mean: {result.max_dd_mean:.2f}% (std: {result.max_dd_std:.2f}%)",
        f"    P5/P50/P95: {result.max_dd_percentiles['5']:.1f}% / "
        f"{result.max_dd_percentiles['50']:.1f}% / "
        f"{result.max_dd_percentiles['95']:.1f}%",
        f"",
        f"  Risk Metrics:",
        f"    VaR 95%: {result.var_95:,.2f}",
        f"    CVaR 95%: {result.cvar_95:,.2f}",
        f"    Prob. of Ruin: {result.probability_of_ruin*100:.2f}%",
        f"",
        f"  Sharpe Ratio:",
        f"    Mean: {result.sharpe_mean:.3f}",
        f"    95% CI: [{result.sharpe_ci_95[0]:.3f}, {result.sharpe_ci_95[1]:.3f}]",
    ]
    return '\n'.join(lines)
