#!/usr/bin/env python3
"""
PF AI Lab 5.6 — Test Suite for v5.6 Features
Tests for all analytics module functions: Omega Ratio, Tail Ratio,
PnL Distribution, Trade Autocorrelation, Exit Reasons, Run-Length,
FTMO Readiness, IS/OOS Degradation, Long/Short, Stability Decay,
OOS Gate Tiers, Enhanced Gate Checks.
"""

import json
import math
import os
import sys
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pf_ma_optimizer.analytics import (
    omega_ratio, tail_ratio,
    analyze_pnl_distribution, trade_autocorrelation,
    metrics_by_exit_reason, run_length_analysis,
    ftmo_readiness, degradation_vector, long_short_analysis,
    stability_decay, oos_gate_tier, enhanced_gate_checks,
    full_v56_analysis,
)


# ============================================================================
# Helper: generate sample trades
# ============================================================================

def _make_trades(pnls, sides=None, exit_reasons=None):
    """Generate a list of trade dicts from PnL values."""
    trades = []
    for i, pnl in enumerate(pnls):
        t = {
            'pnl': pnl,
            's': (sides[i] if sides else ('l' if i % 3 != 0 else 's')),
            'xr': (exit_reasons[i] if exit_reasons else 'setup_revers'),
            'ep': 100.0,
            'xp': 100.0 + pnl,
            'dur': 10 + i,
        }
        trades.append(t)
    return trades


def _trades_json(pnls, **kwargs):
    """Generate JSON string of trades."""
    return json.dumps(_make_trades(pnls, **kwargs))


# ============================================================================
# A1.1: Omega Ratio
# ============================================================================

class TestOmegaRatio:
    """A1.1: Omega Ratio tests."""

    def test_all_positive(self):
        """All positive returns → high omega."""
        r = [1.0, 2.0, 3.0, 0.5]
        assert omega_ratio(r) > 10.0

    def test_all_negative(self):
        """All negative returns → omega near 0."""
        r = [-1.0, -2.0, -3.0]
        assert omega_ratio(r) == 0.0

    def test_mixed_returns(self):
        """Mixed returns: omega between 0 and inf."""
        r = [1.0, -0.5, 0.8, -0.3, 0.6, -0.2]
        o = omega_ratio(r)
        assert 0.0 < o < 999.0
        # Sum of gains > sum of losses → omega > 1
        assert o > 1.0

    def test_symmetric_returns(self):
        """Symmetric returns → omega ≈ 1.0."""
        r = [1.0, -1.0, 1.0, -1.0]
        o = omega_ratio(r)
        assert abs(o - 1.0) < 0.01

    def test_empty_input(self):
        """Empty input → 0."""
        assert omega_ratio([]) == 0.0
        assert omega_ratio(None) == 0.0

    def test_threshold(self):
        """Custom threshold changes result."""
        r = [0.5, 0.3, -0.1, -0.2]
        o_zero = omega_ratio(r, threshold=0.0)
        o_high = omega_ratio(r, threshold=0.3)
        # Higher threshold → lower omega (more below threshold)
        assert o_high < o_zero

    def test_numpy_array(self):
        """Works with numpy arrays."""
        r = np.array([1.0, -0.5, 0.8, -0.3])
        o = omega_ratio(r)
        assert o > 0


# ============================================================================
# A1.2: Tail Ratio
# ============================================================================

class TestTailRatio:
    """A1.2: Tail Ratio tests."""

    def test_symmetric_tails(self):
        """Symmetric distribution → tail ratio ≈ 1.0."""
        np.random.seed(42)
        r = np.random.normal(0, 1, 200)
        tr = tail_ratio(r)
        assert 0.5 < tr < 2.0  # roughly symmetric

    def test_positive_skew(self):
        """Positive skew (big gains, small losses) → tail > 1.0."""
        # Mostly small losses, few big gains
        r = [-0.5] * 80 + [5.0] * 20
        tr = tail_ratio(r)
        assert tr > 1.0

    def test_negative_skew(self):
        """Negative skew (small gains, big losses) → tail < 1.0."""
        r = [0.5] * 80 + [-5.0] * 20
        tr = tail_ratio(r)
        assert tr < 1.0

    def test_too_few_returns(self):
        """Less than 5 returns → 0."""
        assert tail_ratio([1, 2, 3]) == 0.0
        assert tail_ratio(None) == 0.0

    def test_in_metrics(self):
        """Omega and tail_ratio should appear in compute_metrics output."""
        from pf_ma_optimizer.metrics import _empty_metrics
        em = _empty_metrics()
        assert 'omega_ratio' in em
        assert 'tail_ratio' in em


# ============================================================================
# B2.1: PnL Distribution Analysis
# ============================================================================

class TestPnLDistribution:
    """B2.1: PnL distribution analysis."""

    def test_basic_distribution(self):
        """Normal distribution analysis returns all expected keys."""
        pnls = [0.5, -0.3, 0.8, -0.1, 0.3, -0.5, 0.6, -0.2, 0.4, -0.4]
        result = analyze_pnl_distribution(_trades_json(pnls))
        assert result is not None
        assert 'mean' in result
        assert 'std' in result
        assert 'skew' in result
        assert 'kurtosis' in result
        assert 'outliers_2std' in result
        assert 'top5_contribution_pct' in result
        assert 'omega' in result
        assert 'tail_ratio' in result
        assert result['n_trades'] == 10

    def test_empty_trades(self):
        """Empty trades → None."""
        assert analyze_pnl_distribution(None) is None
        assert analyze_pnl_distribution('[]') is None

    def test_positive_mean(self):
        """Mostly winning trades → positive mean."""
        pnls = [1.0, 0.5, -0.2, 0.8, -0.1, 0.6]
        result = analyze_pnl_distribution(_trades_json(pnls))
        assert result['mean'] > 0

    def test_outliers_detection(self):
        """Should detect outliers in distribution."""
        pnls = [0.1] * 50 + [10.0]  # one big outlier
        result = analyze_pnl_distribution(_trades_json(pnls))
        assert result['outliers_2std'] >= 1


# ============================================================================
# B2.2: Trade Autocorrelation
# ============================================================================

class TestTradeAutocorrelation:
    """B2.2: Trade autocorrelation analysis."""

    def test_basic_autocorrelation(self):
        """Should return lag_1 through lag_5 and cluster_risk."""
        pnls = [0.5, -0.3, 0.8, -0.1, 0.3, -0.5, 0.6, -0.2, 0.4, -0.4, 0.2, -0.1]
        result = trade_autocorrelation(_trades_json(pnls))
        assert result is not None
        assert 'lag_1' in result
        assert 'lag_5' in result
        assert 'cluster_risk' in result

    def test_alternating_pattern(self):
        """Alternating win/loss → negative lag-1 autocorrelation."""
        pnls = [1, -1] * 30
        result = trade_autocorrelation(_trades_json(pnls))
        assert result is not None
        assert result['lag_1'] < 0

    def test_clustered_pattern(self):
        """Clustered wins/losses → positive autocorrelation."""
        pnls = [1] * 15 + [-1] * 15 + [1] * 15
        result = trade_autocorrelation(_trades_json(pnls))
        assert result is not None
        assert result['lag_1'] > 0.1

    def test_insufficient_trades(self):
        """Too few trades → None."""
        assert trade_autocorrelation(_trades_json([1, -1])) is None


# ============================================================================
# B2.4: Metrics by Exit Reason
# ============================================================================

class TestMetricsByExitReason:
    """B2.4: Metrics breakdown by exit reason."""

    def test_single_exit_reason(self):
        """All trades same exit reason."""
        pnls = [0.5, -0.3, 0.2]
        result = metrics_by_exit_reason(_trades_json(pnls))
        assert result is not None
        assert 'setup_revers' in result
        assert result['setup_revers']['count'] == 3

    def test_multiple_exit_reasons(self):
        """Multiple exit reasons should be separated."""
        trades = [
            {'pnl': 0.5, 'xr': 'setup_revers'},
            {'pnl': -0.3, 'xr': 'max_hold_day'},
            {'pnl': 0.2, 'xr': 'setup_revers'},
            {'pnl': -0.1, 'xr': 'max_hold_day'},
        ]
        result = metrics_by_exit_reason(json.dumps(trades))
        assert 'setup_revers' in result
        assert 'max_hold_day' in result
        assert result['setup_revers']['count'] == 2
        assert result['max_hold_day']['count'] == 2

    def test_win_rate_by_reason(self):
        """Win rate calculated correctly per reason."""
        trades = [
            {'pnl': 0.5, 'xr': 'setup_revers'},
            {'pnl': 0.3, 'xr': 'setup_revers'},
            {'pnl': -0.1, 'xr': 'max_hold_day'},
            {'pnl': -0.2, 'xr': 'max_hold_day'},
        ]
        result = metrics_by_exit_reason(json.dumps(trades))
        assert result['setup_revers']['win_rate'] == 100.0
        assert result['max_hold_day']['win_rate'] == 0.0


# ============================================================================
# B2.5: Run-Length Analysis
# ============================================================================

class TestRunLengthAnalysis:
    """B2.5: Consecutive win/loss streak analysis."""

    def test_basic_streaks(self):
        """Detect win and loss streaks."""
        pnls = [1, 1, 1, -1, -1, 1, -1, -1, -1, -1]
        result = run_length_analysis(_trades_json(pnls))
        assert result is not None
        assert result['max_win_streak'] == 3
        assert result['max_loss_streak'] == 4

    def test_all_wins(self):
        """All wins → one big win streak, no loss streaks."""
        pnls = [1, 1, 1, 1, 1]
        result = run_length_analysis(_trades_json(pnls))
        assert result['max_win_streak'] == 5
        assert result['max_loss_streak'] == 0

    def test_alternating(self):
        """Alternating → max streak = 1."""
        pnls = [1, -1, 1, -1, 1, -1]
        result = run_length_analysis(_trades_json(pnls))
        assert result['max_win_streak'] == 1
        assert result['max_loss_streak'] == 1


# ============================================================================
# B5.2: FTMO Readiness
# ============================================================================

class TestFTMOReadiness:
    """B5.2: FTMO readiness score."""

    def test_perfect_config(self):
        """Config meeting all FTMO criteria → high score."""
        config = {
            'oos_max_dd': -3.0,
            'oos_net_return': 15.0,
            'oos_expectancy': 0.1,
            'oos_trades': 60,
            'oos_avg_duration': 50,
            'oos_trades_json': _trades_json([0.5, -0.2] * 30),
        }
        result = ftmo_readiness(config, oos_days=30)
        assert result['ftmo_score'] >= 70
        assert result['dd_ok'] is True
        assert result['profit_ok'] is True

    def test_failing_config(self):
        """Config failing all criteria → low score."""
        config = {
            'oos_max_dd': -15.0,
            'oos_net_return': 2.0,
            'oos_expectancy': 0.001,
            'oos_trades': 5,
            'oos_avg_duration': 300,
            'oos_trades_json': _trades_json([-1, -1, -1, -1, -1]),
        }
        result = ftmo_readiness(config, oos_days=30)
        assert result['ftmo_score'] < 40
        assert result['dd_ok'] is False

    def test_consecutive_loss_check(self):
        """Max consecutive losses > 5 → consec_loss_ok = False."""
        pnls = [1, -1, -1, -1, -1, -1, -1, 1, 1]
        config = {
            'oos_max_dd': -4.0,
            'oos_net_return': 12.0,
            'oos_expectancy': 0.05,
            'oos_trades': 9,
            'oos_avg_duration': 50,
            'oos_trades_json': _trades_json(pnls),
        }
        result = ftmo_readiness(config, oos_days=30)
        assert result['consec_loss_ok'] is False
        assert result['max_consec_loss'] == 6

    def test_ftmo_grades(self):
        """Verify grade assignment."""
        config_good = {
            'oos_max_dd': -2.0, 'oos_net_return': 20.0,
            'oos_expectancy': 0.15, 'oos_trades': 100,
            'oos_avg_duration': 30,
            'oos_trades_json': _trades_json([0.3, -0.1] * 50),
        }
        r = ftmo_readiness(config_good, oos_days=20)
        assert r['ftmo_grade'] in ('READY', 'PROMISING')


# ============================================================================
# B5.1: IS/OOS Degradation Vector
# ============================================================================

class TestDegradationVector:
    """B5.1: IS/OOS degradation vector."""

    def test_uniform_degradation(self):
        """Uniform ~30% degradation → UNIFORM symmetry."""
        config = {
            'is_calmar': 2.0, 'oos_calmar': 1.4,
            'is_profit_factor': 2.0, 'oos_profit_factor': 1.4,
            'is_win_rate': 50, 'oos_win_rate': 35,
            'is_trades': 100, 'oos_trades': 30,
            'is_sharpe': 2.0, 'oos_sharpe': 1.4,
        }
        result = degradation_vector(config)
        assert result is not None
        assert result['symmetry'] == 'UNIFORM'

    def test_asymmetric_degradation(self):
        """Calmar collapses but WR stable → ASYMMETRIC."""
        config = {
            'is_calmar': 3.0, 'oos_calmar': 0.3,  # 90% drop
            'is_profit_factor': 2.0, 'oos_profit_factor': 1.8,  # 10% drop
            'is_win_rate': 50, 'oos_win_rate': 48,  # 4% drop
            'is_trades': 100, 'oos_trades': 30,
            'is_sharpe': 2.5, 'oos_sharpe': 2.3,
        }
        result = degradation_vector(config)
        assert result['symmetry'] == 'ASYMMETRIC'

    def test_zero_is_values(self):
        """IS = 0 → degradation = 0 (no division by zero)."""
        config = {
            'is_calmar': 0, 'oos_calmar': 1.0,
            'is_profit_factor': 0, 'oos_profit_factor': 1.5,
            'is_win_rate': 0, 'oos_win_rate': 40,
            'is_trades': 0, 'oos_trades': 20,
            'is_sharpe': 0, 'oos_sharpe': 1.0,
        }
        result = degradation_vector(config)
        assert result['calmar_deg'] == 0.0


# ============================================================================
# B1.5: Long/Short Ratio
# ============================================================================

class TestLongShortAnalysis:
    """B1.5: Long/short ratio warning."""

    def test_balanced(self):
        """50/50 → BALANCED."""
        sides = ['l'] * 10 + ['s'] * 10
        pnls = [0.1] * 20
        result = long_short_analysis(_trades_json(pnls, sides=sides))
        assert result is not None
        assert result['bias_warning'] == 'BALANCED'

    def test_extreme_long_bias(self):
        """95% long → EXTREME_BIAS."""
        sides = ['l'] * 19 + ['s'] * 1
        pnls = [0.1] * 20
        result = long_short_analysis(_trades_json(pnls, sides=sides))
        assert result['bias_warning'] == 'EXTREME_BIAS'
        assert result['long_pct'] == 95.0

    def test_moderate_bias(self):
        """80% long → MODERATE_BIAS."""
        sides = ['l'] * 16 + ['s'] * 4
        pnls = [0.1] * 20
        result = long_short_analysis(_trades_json(pnls, sides=sides))
        assert result['bias_warning'] == 'MODERATE_BIAS'


# ============================================================================
# B5.3: Stability Decay
# ============================================================================

class TestStabilityDecay:
    """B5.3: Stability decay index."""

    def test_stable_performance(self):
        """Consistent performance → STABLE."""
        pnls = [0.3, -0.1] * 20  # alternating, consistent
        result = stability_decay(_trades_json(pnls))
        assert result is not None
        assert result['decay_status'] in ('STABLE', 'MIXED')

    def test_decaying_performance(self):
        """Performance degrading Q1→Q4 → DECAYING or DECLINING."""
        pnls = ([1.0] * 10 + [0.5] * 10 + [0.1] * 10 + [-0.5] * 10)
        result = stability_decay(_trades_json(pnls))
        assert result is not None
        assert result['decay_status'] in ('DECAYING', 'DECLINING')
        assert result['wr_decay'] > 0  # Q1 better than Q4

    def test_insufficient_trades(self):
        """< 8 trades → None."""
        assert stability_decay(_trades_json([1, -1, 1])) is None

    def test_quartile_counts(self):
        """4 quartiles with correct trade counts."""
        pnls = [0.1] * 40
        result = stability_decay(_trades_json(pnls))
        assert len(result['quartiles']) == 4
        assert result['quartiles']['Q1']['n_trades'] == 10


# ============================================================================
# A3.1: OOS Gate Tiers
# ============================================================================

class TestOOSGateTier:
    """A3.1: Multi-level OOS gate classification."""

    def test_gold_tier(self):
        """Config meeting all GOLD criteria."""
        config = {
            'oos_calmar': 1.5,
            'oos_profit_factor': 2.0,
            'oos_trades': 50,
            'perturbation_stable': True,
            'wfe': 0.8,
        }
        result = oos_gate_tier(config)
        assert result['tier'] == 'GOLD'
        assert result['tier_score'] == 3

    def test_silver_tier(self):
        """Config meeting SILVER but not GOLD."""
        config = {
            'oos_calmar': 0.7,
            'oos_profit_factor': 1.4,
            'oos_trades': 30,
            'perturbation_stable': False,  # fails GOLD
            'wfe': 0.3,  # fails GOLD
        }
        result = oos_gate_tier(config)
        assert result['tier'] == 'SILVER'

    def test_bronze_tier(self):
        """Config meeting BRONZE but not SILVER."""
        config = {
            'oos_calmar': 0.30,
            'oos_profit_factor': 1.22,
            'oos_trades': 18,  # < 25 for SILVER
        }
        result = oos_gate_tier(config)
        assert result['tier'] == 'BRONZE'

    def test_fail_tier(self):
        """Config failing all tiers."""
        config = {
            'oos_calmar': 0.1,
            'oos_profit_factor': 1.0,
            'oos_trades': 10,
        }
        result = oos_gate_tier(config)
        assert result['tier'] == 'FAIL'
        assert result['tier_score'] == 0

    def test_gold_requires_all_conditions(self):
        """GOLD requires perturbation_stable=True."""
        config = {
            'oos_calmar': 2.0,
            'oos_profit_factor': 2.5,
            'oos_trades': 60,
            'perturbation_stable': False,  # this alone drops from GOLD
            'wfe': 0.9,
        }
        result = oos_gate_tier(config)
        assert result['tier'] == 'SILVER'  # falls to SILVER


# ============================================================================
# B1.2 + B1.3: Enhanced Gate Checks
# ============================================================================

class TestEnhancedGateChecks:
    """B1.2/B1.3: Enhanced gate using underexploited fields."""

    def test_good_config(self):
        """Good RR ratio + high expectancy → pass."""
        config = {
            'oos_avg_win': 0.5,
            'oos_avg_loss': -0.3,
            'oos_win_rate': 55.0,
            'oos_expectancy': 0.05,
            'oos_avg_duration': 100,
        }
        result = enhanced_gate_checks(config)
        assert result['enhanced_pass'] is True
        assert result['rr_ok'] is True
        assert result['expectancy_ok'] is True

    def test_losing_math(self):
        """WR < 50% and RR < 1.0 → mathematically losing → fail."""
        config = {
            'oos_avg_win': 0.3,
            'oos_avg_loss': -0.5,
            'oos_win_rate': 40.0,
            'oos_expectancy': -0.01,
            'oos_avg_duration': 100,
        }
        result = enhanced_gate_checks(config)
        assert result['rr_ok'] is False
        assert result['enhanced_pass'] is False
        assert len(result['reasons']) >= 1

    def test_low_expectancy(self):
        """Expectancy < 0.02 → fail."""
        config = {
            'oos_avg_win': 0.5,
            'oos_avg_loss': -0.3,
            'oos_win_rate': 55.0,
            'oos_expectancy': 0.01,
            'oos_avg_duration': 100,
        }
        result = enhanced_gate_checks(config)
        assert result['expectancy_ok'] is False


# ============================================================================
# Full v5.6 Analysis
# ============================================================================

class TestFullV56Analysis:
    """Integration test: full_v56_analysis runs all analytics."""

    def test_full_analysis_with_trades(self):
        """Full analysis on a config with trades data."""
        pnls = [0.5, -0.3, 0.8, -0.1, 0.3, -0.5, 0.6, -0.2, 0.4, -0.4] * 5
        config = {
            'oos_calmar': 1.2,
            'oos_profit_factor': 1.6,
            'oos_trades': 50,
            'oos_max_dd': -4.0,
            'oos_net_return': 12.0,
            'oos_expectancy': 0.05,
            'oos_avg_win': 0.5,
            'oos_avg_loss': -0.3,
            'oos_win_rate': 55.0,
            'oos_avg_duration': 100,
            'perturbation_stable': True,
            'wfe': 0.7,
            'is_calmar': 2.0,
            'is_profit_factor': 2.0,
            'is_win_rate': 60,
            'is_trades': 120,
            'is_sharpe': 2.5,
            'oos_sharpe': 1.8,
            'oos_trades_json': _trades_json(pnls),
        }
        result = full_v56_analysis(config)

        assert 'tier' in result
        assert result['tier']['tier'] in ('GOLD', 'SILVER', 'BRONZE', 'FAIL')

        assert 'enhanced_gate' in result
        assert 'degradation' in result

        assert 'pnl_distribution' in result
        assert result['pnl_distribution'] is not None

        assert 'autocorrelation' in result
        assert 'exit_reasons' in result
        assert 'run_length' in result
        assert 'long_short' in result
        assert 'stability' in result
        assert 'ftmo' in result

    def test_full_analysis_without_trades(self):
        """Full analysis on a config without trades data."""
        config = {
            'oos_calmar': 0.5,
            'oos_profit_factor': 1.3,
            'oos_trades': 20,
            'oos_max_dd': -6.0,
            'is_calmar': 1.0,
            'is_profit_factor': 1.8,
            'is_win_rate': 50,
            'is_trades': 60,
            'is_sharpe': 1.5,
            'oos_sharpe': 1.0,
        }
        result = full_v56_analysis(config)

        assert 'tier' in result
        assert 'enhanced_gate' in result
        assert 'degradation' in result
        # Trade-level analytics should not be present
        assert 'pnl_distribution' not in result
        assert 'ftmo' not in result


# ============================================================================
# Metrics.py: Omega + Tail in compute_metrics output
# ============================================================================

class TestMetricsOmegaTail:
    """Verify omega_ratio and tail_ratio are in compute_metrics output."""

    def test_empty_metrics_has_omega_tail(self):
        """_empty_metrics() includes omega and tail ratio."""
        from pf_ma_optimizer.metrics import _empty_metrics
        em = _empty_metrics()
        assert 'omega_ratio' in em
        assert 'tail_ratio' in em
        assert em['omega_ratio'] == 0
        assert em['tail_ratio'] == 0

    def test_config_has_tier_thresholds(self):
        """Config module has OOS_GATE_TIERS."""
        from pf_ma_optimizer.config import OOS_GATE_TIERS
        assert 'GOLD' in OOS_GATE_TIERS
        assert 'SILVER' in OOS_GATE_TIERS
        assert 'BRONZE' in OOS_GATE_TIERS
        assert OOS_GATE_TIERS['GOLD']['min_calmar'] > OOS_GATE_TIERS['SILVER']['min_calmar']
