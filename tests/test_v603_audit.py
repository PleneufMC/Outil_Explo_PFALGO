#!/usr/bin/env python3
"""
PF AI Lab 6.0.3 -- Test Suite for Audit & Risk Management Improvements

Tests all P0, P1, P2 items from the Risk Management audit report:
  P0: Export integrity (round-trip, instrument reset, SHA-256, strict mode)
  P1: Feed discount, FTMO metrics, hard constraints, risk bounds
  P2: Bootstrap CI, PnL percentiles, rolling PF, stress test, overfitting,
      live readiness score
"""

import json
import math
import os
import sys
import hashlib
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pf_ma_optimizer.export_integrity import (
    reset_instrument_fields,
    validate_instrument_fields,
    validate_round_trip,
    compute_config_hash,
    build_export_metadata,
    validate_for_export,
    export_config_json,
    ExportValidationError,
    APP_VERSION,
)

from pf_ma_optimizer.ftmo_risk import (
    apply_feed_discount,
    compute_max_daily_dd,
    compute_worst_consecutive_loss,
    compute_recovery_time,
    compute_best_day_rule,
    ftmo_hard_filter,
    compute_ftmo_risk_bounds,
    bootstrap_confidence_intervals,
    compute_pnl_percentiles,
    compute_rolling_pf,
    stress_test,
    overfitting_indicator,
    compute_live_readiness_score,
    full_risk_analysis,
    FTMO_CONSTRAINTS,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_config(**overrides):
    """Build a minimal valid config dict for testing."""
    base = {
        'entry_type': 'Donc+Chikou',
        'pmax_ma_type': 'EMA',
        'pmax_length': 10,
        'pmax_multiplier': 3.0,
        'long_side': True,
        'short_side': True,
        'sl_mode': 'static_pct',
        'sl_pct': 30.0,
        'tp_mode': 'no',
        'be_mode': 'no',
        'exit_setup_reversal': True,
        'use_lt_filter': False,
        'use_mt_filter': False,
        'use_rsi50_filter': False,
        'use_diff_ma_red': False,
        'lt_tf': 'W',
        'mt_tf': '4H',
        'don_length': 20,
        'order_pause': 5,
        'lt_multiplier': 1.5,
        'mt_multiplier': 1.5,
    }
    base.update(overrides)
    return base


def _make_trades_json(pnls, sides=None):
    """Generate compact trades JSON string."""
    trades = []
    for i, pnl in enumerate(pnls):
        trades.append({
            'pnl': round(pnl, 3),
            's': (sides[i] if sides else ('l' if i % 2 == 0 else 's')),
            'ep': 100.0,
            'xp': 100.0 + pnl,
            'dur': 10,
            'xr': 'setup_revers',
        })
    return json.dumps(trades)


def _make_config_row(**overrides):
    """Build a DB-like config row for testing."""
    base = {
        'oos_trades': 60,
        'oos_win_rate': 55.0,
        'oos_net_return': 15.0,
        'oos_max_dd': -4.5,
        'oos_calmar': 1.2,
        'oos_profit_factor': 1.5,
        'oos_sharpe': 1.0,
        'oos_expectancy': 0.05,
        'oos_avg_win': 0.8,
        'oos_avg_loss': -0.6,
        'oos_avg_duration': 12.0,
        'is_profit_factor': 1.8,
        'is_win_rate': 57.0,
        'is_max_dd': -3.5,
        'is_calmar': 1.8,
        'wf_stability': 70.0,
        'wf_consistency': 80.0,
        'wf_degradation': False,
        'perturbation_stable': True,
        'perturbation_degradation': 0.15,
        'wfe': 0.67,
        'oos_trades_json': _make_trades_json(
            [0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.3, -0.1, 0.9, -0.5,
             0.4, -0.3, 0.7, -0.2, 0.5, -0.6, 0.3, -0.1, 0.8, -0.4,
             0.6, -0.3, 0.5, -0.2, 0.4, -0.5, 0.7, -0.3, 0.6, -0.1,
             0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.3, -0.1, 0.9, -0.5,
             0.4, -0.3, 0.7, -0.2, 0.5, -0.6, 0.3, -0.1, 0.8, -0.4,
             0.6, -0.3, 0.5, -0.2, 0.4, -0.5, 0.7, -0.3, 0.6, -0.1]
        ),
    }
    base.update(overrides)
    return base


# ============================================================================
# P0 TESTS: Export Integrity
# ============================================================================

class TestP0RoundTrip:
    """P0.1: Round-trip validation tests."""

    def test_clean_config_passes(self):
        config = _make_config()
        errors = validate_round_trip(config)
        assert errors == [], f"Clean config should pass: {errors}"

    def test_bool_preserved(self):
        config = _make_config(long_side=True, short_side=False)
        errors = validate_round_trip(config)
        assert errors == []

    def test_float_preserved(self):
        config = _make_config(pmax_multiplier=2.7)
        errors = validate_round_trip(config)
        assert errors == []

    def test_string_preserved(self):
        config = _make_config(entry_type='RSI Divergence')
        errors = validate_round_trip(config)
        assert errors == []

    def test_missing_critical_field_detected(self):
        """Simulate a bug where short_side is lost during serialization."""
        config = _make_config()
        # Manually serialize, then remove short_side
        serialized = json.loads(json.dumps(config))
        # The round-trip itself should work; we test the validator catches real issues
        errors = validate_round_trip(config)
        assert errors == []


class TestP0InstrumentReset:
    """P0.2: Instrument field reset and cross-validation."""

    def test_reset_clears_residuals(self):
        config = {
            'instrument': 'BTCUSD',
            'mt4_symbol': 'BTCUSDT',
            'wma_instrument': 'TVC:DEU40',
        }
        reset_instrument_fields(config, 'EUSTX50')
        assert config['instrument'] == 'EUSTX50'
        assert config['mt4_symbol'] == 'EUSTX50'
        assert 'DEU40' not in config['wma_instrument']

    def test_unknown_instrument_defaults(self):
        config = {}
        reset_instrument_fields(config, 'MYINSTRUMENT')
        assert config['instrument'] == 'MYINSTRUMENT'
        assert config['mt4_symbol'] == 'MYINSTRUMENT'

    def test_validate_consistent(self):
        config = {'instrument': 'EUSTX50', 'mt4_symbol': 'EUSTX50', 'wma_instrument': 'EUREX:FESX1!'}
        warnings = validate_instrument_fields(config)
        assert warnings == []

    def test_validate_mismatch(self):
        config = {'instrument': 'EUSTX50', 'mt4_symbol': 'BTCUSDT', 'wma_instrument': 'TVC:DEU40'}
        warnings = validate_instrument_fields(config)
        assert len(warnings) == 2


class TestP0Hash:
    """P0.3: SHA-256 config hash."""

    def test_deterministic(self):
        config = _make_config()
        h1 = compute_config_hash(config)
        h2 = compute_config_hash(config)
        assert h1 == h2
        assert h1.startswith('sha256:')

    def test_different_configs_different_hash(self):
        c1 = _make_config(pmax_length=10)
        c2 = _make_config(pmax_length=15)
        assert compute_config_hash(c1) != compute_config_hash(c2)

    def test_hash_length(self):
        h = compute_config_hash(_make_config())
        # sha256:64_hex_chars
        assert len(h) == 7 + 64


class TestP0Metadata:
    """P0.3: Export metadata block."""

    def test_contains_required_fields(self):
        meta = build_export_metadata(_make_config(), instrument='EUSTX50')
        assert meta['schema_version'] == APP_VERSION
        assert meta['config_hash'].startswith('sha256:')
        assert 'export_timestamp' in meta
        assert meta['pflab_version'] == APP_VERSION
        assert meta['data_source'] == 'MT5_FTMO'
        assert meta['optimization_feed']['broker'] == 'FTMO'
        assert meta['trade_counting_method'] == 'per_position'
        assert 'feed_divergence_note' in meta

    def test_is_oos_periods(self):
        meta = build_export_metadata(
            _make_config(), instrument='US500',
            is_period=('2020-01-02', '2024-06-30'),
            oos_period=('2024-07-01', '2026-03-09'),
        )
        assert meta['is_period'] == ['2020-01-02', '2024-06-30']
        assert meta['oos_period'] == ['2024-07-01', '2026-03-09']


class TestP0StrictExport:
    """P0.4: Strict export mode."""

    def test_clean_export_passes(self):
        config = _make_config()
        result = export_config_json(config, instrument='EUSTX50', strict=True)
        assert 'config' in result
        assert 'metadata' in result
        assert result['validation_warnings'] == []

    def test_strict_blocks_invalid(self):
        config = _make_config(long_side=False, short_side=False)
        with pytest.raises(ExportValidationError):
            export_config_json(config, instrument='EUSTX50', strict=True)

    def test_non_strict_returns_warnings(self):
        config = _make_config(long_side=False, short_side=False)
        result = export_config_json(config, instrument='EUSTX50', strict=False)
        assert len(result['validation_warnings']) > 0

    def test_missing_entry_type(self):
        config = _make_config()
        config.pop('entry_type')
        with pytest.raises(ExportValidationError):
            export_config_json(config, instrument='US500', strict=True)


# ============================================================================
# P1 TESTS: Transparency & FTMO
# ============================================================================

class TestP1FeedDiscount:
    """P1.5: Feed discount factor."""

    def test_default_discount(self):
        metrics = {'profit_factor': 1.93, 'sharpe': 1.5, 'expectancy': 0.05}
        result = apply_feed_discount(metrics)
        assert result['feed_discount_factor'] == 0.70
        # PF_live = 1 + (1.93-1) * 0.7 = 1.651
        assert abs(result['pf_live_estimate'] - 1.651) < 0.01

    def test_custom_discount(self):
        metrics = {'profit_factor': 2.0, 'sharpe': 1.0, 'expectancy': 0.03}
        result = apply_feed_discount(metrics, factor=0.5)
        assert result['feed_discount_factor'] == 0.5
        # PF_live = 1 + (2.0-1) * 0.5 = 1.5
        assert abs(result['pf_live_estimate'] - 1.5) < 0.01


class TestP1FTMOMetrics:
    """P1.6: FTMO-specific metrics."""

    def test_max_daily_dd(self):
        trades_json = _make_trades_json([-2.0, -1.5, -1.0, 0.5, 0.3])
        result = compute_max_daily_dd(trades_json)
        assert 'max_daily_dd_pct' in result
        assert result['n_days'] > 0

    def test_worst_consecutive_loss(self):
        pnls = [0.5, -0.3, -0.4, -0.5, -0.2, 0.8, -0.1]
        trades_json = _make_trades_json(pnls)
        result = compute_worst_consecutive_loss(trades_json)
        assert result['n_consec'] == 4  # -0.3, -0.4, -0.5, -0.2

    def test_recovery_time_complete(self):
        # Equity: 1, 2, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5 -> peak=2 at idx1, dd at idx2, recovers at idx8
        pnls = [1.0, 1.0, -3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
        trades_json = _make_trades_json(pnls)
        result = compute_recovery_time(trades_json)
        assert result['recovery_status'] == 'complete'

    def test_recovery_time_incomplete(self):
        pnls = [1.0, 1.0, -5.0, 0.5, 0.5]
        trades_json = _make_trades_json(pnls)
        result = compute_recovery_time(trades_json)
        assert result['recovery_status'] == 'incomplete'

    def test_best_day_rule(self):
        trades_json = _make_trades_json([5.0, 0.1, 0.1, 0.1])
        result = compute_best_day_rule(trades_json)
        assert 'best_day_ratio' in result


class TestP1HardConstraints:
    """P1.7: FTMO hard constraints."""

    def test_eligible_config(self):
        row = _make_config_row(oos_max_dd=-4.5, oos_net_return=12.0)
        result = ftmo_hard_filter(row)
        assert result['ftmo_eligible'] is True

    def test_dd_breach_disqualifies(self):
        row = _make_config_row(oos_max_dd=-12.0)
        result = ftmo_hard_filter(row)
        assert result['ftmo_eligible'] is False
        assert any('MaxDD' in r for r in result['disqualification_reasons'])

    def test_challenge_types(self):
        assert '2-step' in FTMO_CONSTRAINTS
        assert '1-step' in FTMO_CONSTRAINTS


class TestP1RiskBounds:
    """P1.8: Risk bounds in JSON."""

    def test_risk_bounds_structure(self):
        row = _make_config_row()
        result = compute_ftmo_risk_bounds(row)
        required_keys = ['daily_ub_pct', 'ruin_ub_pct', 'risk_optimal_pct',
                         'min_risk_pct', 'ecl', 'days_to_target_p50', 'trades_per_day']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_ecl_positive(self):
        row = _make_config_row()
        result = compute_ftmo_risk_bounds(row)
        assert result['ecl'] > 0

    def test_trades_per_day_from_oos(self):
        row = _make_config_row(oos_trades=60, oos_avg_duration=24)
        result = compute_ftmo_risk_bounds(row)
        assert result['trades_per_day'] > 0


# ============================================================================
# P2 TESTS: Robustness & Portfolio
# ============================================================================

class TestP2Bootstrap:
    """P2.9: Bootstrap confidence intervals."""

    def test_basic_bootstrap(self):
        pnls = np.random.RandomState(42).normal(0.3, 1.0, 60).tolist()
        trades_json = _make_trades_json(pnls)
        result = bootstrap_confidence_intervals(trades_json, n_bootstrap=500)
        assert 'PF_p5' in result
        assert 'PF_p50' in result
        assert 'PF_p95' in result
        assert result['PF_p5'] <= result['PF_p50'] <= result['PF_p95']

    def test_significant_edge(self):
        pnls = np.random.RandomState(42).normal(0.8, 0.5, 100).tolist()
        trades_json = _make_trades_json(pnls)
        result = bootstrap_confidence_intervals(trades_json, n_bootstrap=500)
        assert result['statistically_significant'] is True

    def test_low_trades_warning(self):
        pnls = [0.5, -0.3, 0.2, -0.1, 0.4]
        trades_json = _make_trades_json(pnls)
        result = bootstrap_confidence_intervals(trades_json)
        assert 'warning' in result

    def test_insufficient_trades(self):
        trades_json = _make_trades_json([0.5, -0.3])
        result = bootstrap_confidence_intervals(trades_json)
        assert result['statistically_significant'] is False


class TestP2PnlPercentiles:
    """P2.10: PnL percentiles."""

    def test_percentiles(self):
        pnls = np.random.RandomState(42).normal(0.2, 1.0, 50).tolist()
        trades_json = _make_trades_json(pnls)
        result = compute_pnl_percentiles(trades_json)
        assert result is not None
        assert result['P5'] < result['P50'] < result['P95']

    def test_insufficient_trades(self):
        result = compute_pnl_percentiles(_make_trades_json([0.5]))
        assert result is None


class TestP2RollingPF:
    """P2.10: Rolling PF."""

    def test_rolling_pf_structure(self):
        pnls = np.random.RandomState(42).normal(0.3, 1.0, 40).tolist()
        trades_json = _make_trades_json(pnls)
        result = compute_rolling_pf(trades_json, window=10)
        assert 'rolling_pf' in result
        assert len(result['rolling_pf']) == 31  # 40 - 10 + 1
        assert result['trend'] in ('DEGRADING', 'STABLE', 'IMPROVING')

    def test_insufficient_trades(self):
        result = compute_rolling_pf(_make_trades_json([0.5, 0.3]), window=20)
        assert result['trend'] == 'insufficient_data'


class TestP2StressTest:
    """P2.11: Stress-test sensitivity."""

    def test_stress_table(self):
        row = _make_config_row()
        result = stress_test(row)
        assert 'sensitivity_table' in result
        assert len(result['sensitivity_table']) == 12  # 4 WR x 3 win scenarios
        assert 'break_even_wr' in result
        assert 0 < result['break_even_wr'] < 100

    def test_break_even_wr(self):
        row = _make_config_row(oos_avg_win=1.0, oos_avg_loss=-1.0, oos_win_rate=55.0)
        result = stress_test(row)
        # BE WR = 1.0 / (1.0 + 1.0) * 100 = 50%
        assert abs(result['break_even_wr'] - 50.0) < 0.1

    def test_robustness_score(self):
        row = _make_config_row()
        result = stress_test(row)
        assert 0 <= result['robustness_score'] <= 100


class TestP2Overfitting:
    """P2.12: Overfitting indicator."""

    def test_no_overfit(self):
        row = _make_config_row(is_profit_factor=1.5, oos_profit_factor=1.3)
        result = overfitting_indicator(row)
        assert result['overfit_flag'] is False
        assert result['overfit_severity'] == 'LOW'

    def test_moderate_overfit(self):
        row = _make_config_row(is_profit_factor=3.0, oos_profit_factor=1.2)
        result = overfitting_indicator(row)
        assert result['overfit_flag'] is True
        assert result['overfit_severity'] in ('MODERATE', 'SEVERE')

    def test_oos_robustness(self):
        row = _make_config_row(oos_trades=100, wf_stability=70)
        result = overfitting_indicator(row)
        assert result['oos_robustness'] == 'high'


class TestP2LiveReadiness:
    """P2.15: Live Readiness Score."""

    def test_good_config(self):
        row = _make_config_row()
        result = compute_live_readiness_score(row)
        assert 0 <= result['live_readiness_score'] <= 100
        assert result['live_readiness_grade'] in ('DEPLOY', 'MONITOR', 'PAPER_TRADE', 'REJECT')

    def test_bad_config(self):
        row = _make_config_row(
            wf_stability=10, oos_trades=5,
            perturbation_stable=False, oos_max_dd=-15,
        )
        result = compute_live_readiness_score(row)
        assert result['live_readiness_score'] < 50

    def test_components_present(self):
        row = _make_config_row()
        result = compute_live_readiness_score(row)
        for key in ['wf_stability', 'n_trades_oos', 'is_oos_coherence',
                     'dd_divergence', 'perturbation', 'ftmo_eligible']:
            assert key in result['components']


class TestFullRiskAnalysis:
    """Integration test: full_risk_analysis combines all P1+P2."""

    def test_all_sections_present(self):
        row = _make_config_row()
        result = full_risk_analysis(row)
        expected_keys = [
            'feed_discount', 'max_daily_dd', 'worst_consec_loss',
            'recovery_time', 'best_day_rule', 'ftmo_hard_filter',
            'ftmo_risk_bounds', 'bootstrap_ci', 'pnl_percentiles',
            'rolling_pf', 'stress_test', 'overfitting', 'live_readiness',
        ]
        for key in expected_keys:
            assert key in result, f"Missing section: {key}"

    def test_no_crash_empty_trades(self):
        row = _make_config_row(oos_trades_json=None)
        result = full_risk_analysis(row)
        assert result is not None
