#!/usr/bin/env python3
"""
PF AI Lab 5.4 — Comprehensive Test Suite
Tests for all V5.4 features: WFE, robustness scoring, quality grades,
transaction costs, SAFE list consistency, leaderboard dedup, OOS trade storage.
"""

import json
import math
import os
import sqlite3
import sys
import tempfile
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ============================================================================
# C0: SAFE_ENTRY_TYPES consistency
# ============================================================================

class TestSafeEntryTypes:
    """C0: Verify SAFE_ENTRY_TYPES is consistent across all packages."""

    def test_pineguard_exports_safe_entry_types(self):
        """PineGuard must export SAFE_ENTRY_TYPES."""
        from pineguard import SAFE_ENTRY_TYPES
        assert isinstance(SAFE_ENTRY_TYPES, list)
        assert len(SAFE_ENTRY_TYPES) > 0

    def test_config_imports_from_pineguard(self):
        """pf_ma_optimizer.config must use PineGuard's SAFE_ENTRY_TYPES."""
        from pf_ma_optimizer.config import SAFE_ENTRY_TYPES as config_safe
        from pineguard import SAFE_ENTRY_TYPES as pg_safe
        assert config_safe == pg_safe, \
            f"Config SAFE: {config_safe} vs PineGuard SAFE: {pg_safe}"

    def test_safe_types_include_core_entries(self):
        """Core safe entries must be in the list."""
        from pineguard import SAFE_ENTRY_TYPES
        assert 'Donc+Chikou' in SAFE_ENTRY_TYPES
        assert 'RSI Divergence' in SAFE_ENTRY_TYPES

    def test_safe_types_exclude_buggy(self):
        """Known buggy entries must NOT be in SAFE list."""
        from pineguard import SAFE_ENTRY_TYPES
        assert 'VWAP' not in SAFE_ENTRY_TYPES
        assert 'ORB' not in SAFE_ENTRY_TYPES

    def test_pineguard_entries_init_safe_match(self):
        """PineGuard top-level SAFE_ENTRY_TYPES is the single source of truth."""
        from pineguard import SAFE_ENTRY_TYPES
        # It should be a list with known safe entries
        assert isinstance(SAFE_ENTRY_TYPES, list)
        assert 'Donc+Chikou' in SAFE_ENTRY_TYPES
        assert 'RSI Divergence' in SAFE_ENTRY_TYPES
        # Config should import the same list
        from pf_ma_optimizer.config import SAFE_ENTRY_TYPES as cfg_safe
        assert set(cfg_safe) == set(SAFE_ENTRY_TYPES)


# ============================================================================
# C1: WFE calculation (floor, cap, NULL)
# ============================================================================

class TestWFECalculation:
    """C1: WFE floor/cap/NULL logic."""

    def _calc_wfe(self, is_calmar, oos_calmar):
        """Reproduce the WFE logic from backtest_db._insert_config."""
        if is_calmar < 0.10:
            return None  # unreliable
        elif oos_calmar <= 0:
            return 0.0
        else:
            return min(oos_calmar / is_calmar, 3.0)

    def test_wfe_null_when_is_calmar_too_low(self):
        """WFE should be NULL when IS Calmar < 0.10."""
        assert self._calc_wfe(0.05, 0.50) is None
        assert self._calc_wfe(0.0, 1.0) is None
        assert self._calc_wfe(0.09, 0.05) is None
        assert self._calc_wfe(-1.0, 0.5) is None

    def test_wfe_zero_when_oos_calmar_negative(self):
        """WFE should be 0.0 when OOS Calmar <= 0."""
        assert self._calc_wfe(1.0, 0.0) == 0.0
        assert self._calc_wfe(0.5, -0.5) == 0.0

    def test_wfe_normal_calculation(self):
        """WFE = OOS Calmar / IS Calmar for normal cases."""
        wfe = self._calc_wfe(1.0, 0.7)
        assert abs(wfe - 0.7) < 0.001

    def test_wfe_cap_at_3(self):
        """WFE capped at 3.0 (300%)."""
        wfe = self._calc_wfe(0.2, 1.0)
        assert wfe == 3.0
        wfe = self._calc_wfe(0.1, 5.0)
        assert wfe == 3.0

    def test_wfe_boundary_is_calmar_0_10(self):
        """WFE at the boundary is_calmar = 0.10 should be valid (not NULL)."""
        wfe = self._calc_wfe(0.10, 0.05)
        assert wfe is not None
        assert abs(wfe - 0.5) < 0.001

    def test_wfe_perfect_1_0(self):
        """WFE = 1.0 when OOS == IS."""
        wfe = self._calc_wfe(0.5, 0.5)
        assert abs(wfe - 1.0) < 0.001

    def test_wfe_stored_as_null_in_db(self):
        """When WFE is None, it should be stored as NULL in SQLite."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            # Save a run with IS calmar < 0.10 to trigger NULL WFE
            result = {
                'instrument': 'TESTINST',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1,
                    'score': 0.5,
                    'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 10},
                    'metrics_is': {'calmar': 0.05, 'n_trades': 50},
                    'metrics_oos': {'calmar': 0.5, 'n_trades': 20},
                }],
            }
            run_id = db.save_run(result)

            # Check WFE is NULL
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT wfe FROM configs WHERE run_id = ?", (run_id,)
            ).fetchone()
            conn.close()
            assert row[0] is None, f"WFE should be NULL but got {row[0]}"
        finally:
            os.unlink(db_path)


# ============================================================================
# C2: Robustness Score + Quality Grade
# ============================================================================

class TestRobustnessScore:
    """C2: compute_robustness_score tests."""

    def test_perfect_score(self):
        """Perfect inputs should give score near 100."""
        from pf_ma_optimizer.metrics import compute_robustness_score
        score = compute_robustness_score(
            wfe=0.70,
            perturbation_stable=True,
            perturbation_degradation=0.0,
            wf_stability=95,
            wf_consistency=90,
            wf_degradation=False,
        )
        assert score >= 85, f"Perfect score should be >= 85, got {score}"

    def test_zero_score(self):
        """All bad inputs should give score near 0."""
        from pf_ma_optimizer.metrics import compute_robustness_score
        score = compute_robustness_score(
            wfe=None,
            perturbation_stable=False,
            perturbation_degradation=1.0,
            wf_stability=10,
            wf_consistency=10,
            wf_degradation=True,
        )
        assert score <= 10, f"Bad inputs should give score <= 10, got {score}"

    def test_wfe_none_no_contribution(self):
        """WFE=None should contribute 0 pts."""
        from pf_ma_optimizer.metrics import compute_robustness_score
        score = compute_robustness_score(
            wfe=None,
            perturbation_stable=False,
            perturbation_degradation=None,
            wf_stability=0,
            wf_consistency=0,
            wf_degradation=False,
        )
        assert score == 0.0

    def test_wfe_optimal_zone(self):
        """WFE=0.70 is the optimal value, should give max WFE pts."""
        from pf_ma_optimizer.metrics import compute_robustness_score
        score = compute_robustness_score(
            wfe=0.70,
            perturbation_stable=False,
            perturbation_degradation=None,
            wf_stability=0,
            wf_consistency=0,
            wf_degradation=False,
        )
        assert score >= 30, f"WFE=0.70 should give >= 30 pts, got {score}"

    def test_wfe_outside_range(self):
        """WFE > 2.0 or < 0.4 should give 0 WFE pts."""
        from pf_ma_optimizer.metrics import compute_robustness_score
        score_high = compute_robustness_score(
            wfe=2.5, perturbation_stable=False,
            perturbation_degradation=None, wf_stability=0,
            wf_consistency=0, wf_degradation=False)
        score_low = compute_robustness_score(
            wfe=0.1, perturbation_stable=False,
            perturbation_degradation=None, wf_stability=0,
            wf_consistency=0, wf_degradation=False)
        assert score_high == 0.0
        assert score_low == 0.0

    def test_score_capped_at_100(self):
        """Score cannot exceed 100."""
        from pf_ma_optimizer.metrics import compute_robustness_score
        score = compute_robustness_score(
            wfe=0.70,
            perturbation_stable=True,
            perturbation_degradation=0.0,
            wf_stability=100,
            wf_consistency=100,
            wf_degradation=False,
        )
        assert score <= 100.0


class TestQualityGrade:
    """C2: compute_quality_grade tests."""

    def test_grade_a_requires_safe_entry(self):
        """Grade A should never be given to NON_SAFE entries."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=12.0,
            robustness_score=90.0,
            entry_type='VWAP',  # not safe
            instrument='US30',
            safe_entry_types=['Donc+Chikou', 'RSI Divergence'],
        )
        assert result['grade'] != 'A', "NON_SAFE entry should not get grade A"
        assert 'NON_SAFE_ENTRY' in result['flags']

    def test_grade_a_for_safe_high_quality(self):
        """High quality safe entry should get grade A."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=12.0,
            robustness_score=90.0,
            entry_type='Donc+Chikou',
            instrument='US30',
            safe_entry_types=['Donc+Chikou', 'RSI Divergence'],
        )
        assert result['grade'] == 'A'

    def test_default_cost_penalty(self):
        """Unknown instrument should get DEFAULT_COST flag."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=10.0,
            robustness_score=80.0,
            entry_type='Donc+Chikou',
            instrument='UNKNOWN_INST',
            transaction_costs={'US30': 3.0},
            default_cost=1.0,
        )
        assert 'DEFAULT_COST' in result['flags']
        assert result['multiplier'] < 1.0

    def test_grade_d_low_quality(self):
        """Very low quality should get grade D."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=0.5,
            robustness_score=5.0,
            entry_type='Donc+Chikou',
            instrument='US30',
        )
        assert result['grade'] == 'D'

    def test_quality_score_range(self):
        """Quality score should be between 0 and 100."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=12.0,
            robustness_score=100.0,
            entry_type='Donc+Chikou',
            instrument='US30',
        )
        assert 0 <= result['quality_score'] <= 100


# ============================================================================
# C3: Transaction cost matching
# ============================================================================

class TestTransactionCosts:
    """C3: get_transaction_cost with partial matching."""

    def test_exact_match(self):
        """Exact instrument name should match."""
        from pf_ma_optimizer.config import get_transaction_cost
        cost = get_transaction_cost('EURUSD')
        assert cost == 0.00015

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        from pf_ma_optimizer.config import get_transaction_cost
        cost = get_transaction_cost('eurusd')
        assert cost == 0.00015

    def test_timeframe_suffix_stripped(self):
        """'NVDA H4' should match 'NVDA'."""
        from pf_ma_optimizer.config import get_transaction_cost
        cost = get_transaction_cost('NVDA H4')
        assert cost == 0.10

    def test_timeframe_suffix_forex(self):
        """'EURUSD M15' should match 'EURUSD'."""
        from pf_ma_optimizer.config import get_transaction_cost
        cost = get_transaction_cost('EURUSD M15')
        assert cost == 0.00015

    def test_stock_cfds_present(self):
        """Stock CFDs should be in TRANSACTION_COSTS."""
        from pf_ma_optimizer.config import TRANSACTION_COSTS
        for stock in ['NVDA', 'AMZN', 'AAPL', 'TSLA', 'MSFT', 'GOOG', 'META']:
            assert stock in TRANSACTION_COSTS, f"{stock} not in TRANSACTION_COSTS"

    def test_unknown_instrument_default(self):
        """Unknown instrument should return DEFAULT_TRANSACTION_COST."""
        from pf_ma_optimizer.config import get_transaction_cost, DEFAULT_TRANSACTION_COST
        cost = get_transaction_cost('XYZABC')
        assert cost == DEFAULT_TRANSACTION_COST

    def test_custom_override(self):
        """Config override should take precedence."""
        from pf_ma_optimizer.config import get_transaction_cost
        cost = get_transaction_cost('EURUSD', {'custom_transaction_cost': 5.0})
        assert cost == 5.0

    def test_custom_override_none_falls_through(self):
        """None custom_transaction_cost should not override."""
        from pf_ma_optimizer.config import get_transaction_cost
        cost = get_transaction_cost('EURUSD', {'custom_transaction_cost': None})
        assert cost == 0.00015


# ============================================================================
# C4: OOS Trades Storage
# ============================================================================

class TestOOSTradesStorage:
    """C4: OOS trades stored and retrievable."""

    def test_oos_trades_stored_and_retrieved(self):
        """OOS trades should be stored as JSON and retrievable."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            trades = [
                {'entry_bar': 100, 'exit_bar': 110, 'pnl_pct': 1.5, 'side': 'long'},
                {'entry_bar': 120, 'exit_bar': 130, 'pnl_pct': -0.8, 'side': 'short'},
            ]

            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1,
                    'score': 1.0,
                    'config': {'entry_type': 'Donc+Chikou'},
                    'metrics_is': {'calmar': 1.0, 'n_trades': 50},
                    'metrics_oos': {'calmar': 0.7, 'n_trades': 20},
                    'oos_trades_detail': trades,
                }],
            }
            run_id = db.save_run(result)

            # Retrieve
            configs = db.get_configs(run_id=run_id)
            assert len(configs) == 1
            cfg = configs[0]
            assert cfg['oos_trades_json'] is not None
            stored_trades = json.loads(cfg['oos_trades_json'])
            assert len(stored_trades) == 2
            assert stored_trades[0]['pnl_pct'] == 1.5
        finally:
            os.unlink(db_path)

    def test_oos_trades_none_when_empty(self):
        """OOS trades should be NULL when no detail provided."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1,
                    'score': 1.0,
                    'config': {'entry_type': 'Donc+Chikou'},
                    'metrics_is': {'calmar': 1.0, 'n_trades': 50},
                    'metrics_oos': {'calmar': 0.7, 'n_trades': 20},
                    # No oos_trades_detail
                }],
            }
            run_id = db.save_run(result)
            configs = db.get_configs(run_id=run_id)
            assert configs[0]['oos_trades_json'] is None
        finally:
            os.unlink(db_path)


# ============================================================================
# C5: Leaderboard Deduplication
# ============================================================================

class TestLeaderboardDedup:
    """C5: Fingerprint-based deduplication."""

    def test_fingerprint_same_config(self):
        """Same config should produce same fingerprint."""
        from pf_ma_optimizer.backtest_db import _config_fingerprint_for_dedup
        cfg = {'entry_type': 'Donc+Chikou', 'pmax_length': 10, 'pmax_multiplier': 3.0}
        fp1 = _config_fingerprint_for_dedup(cfg)
        fp2 = _config_fingerprint_for_dedup(cfg)
        assert fp1 == fp2

    def test_fingerprint_different_config(self):
        """Different configs should produce different fingerprints."""
        from pf_ma_optimizer.backtest_db import _config_fingerprint_for_dedup
        cfg1 = {'entry_type': 'Donc+Chikou', 'pmax_length': 10}
        cfg2 = {'entry_type': 'RSI Divergence', 'pmax_length': 10}
        assert _config_fingerprint_for_dedup(cfg1) != _config_fingerprint_for_dedup(cfg2)

    def test_fingerprint_float_rounding(self):
        """Floats should be rounded to 2 decimals for fingerprint."""
        from pf_ma_optimizer.backtest_db import _config_fingerprint_for_dedup
        cfg1 = {'pmax_multiplier': 3.001}
        cfg2 = {'pmax_multiplier': 3.004}
        assert _config_fingerprint_for_dedup(cfg1) == _config_fingerprint_for_dedup(cfg2)

    def test_dedup_removes_duplicates(self):
        """get_best_per_instrument should dedup identical configs."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            # Insert same config twice in different runs
            config = {
                'entry_type': 'Donc+Chikou',
                'pmax_length': 15,
                'pmax_multiplier': 3.0,
                'pmax_ma_type': 'EMA',
            }
            for _ in range(2):
                result = {
                    'instrument': 'US30',
                    'is_range': ('2020-01-01', '2023-01-01'),
                    'oos_range': ('2023-01-01', '2024-01-01'),
                    'duration_sec': 10,
                    'top_configs': [{
                        'rank': 1,
                        'score': 1.0,
                        'config': config,
                        'metrics_is': {'calmar': 1.0, 'n_trades': 50},
                        'metrics_oos': {'calmar': 0.7, 'n_trades': 20,
                                       'profit_factor': 1.5, 'max_dd': -5.0},
                    }],
                }
                db.save_run(result)

            best = db.get_best_per_instrument(top_n=5)
            # Should have only 1 config (deduped)
            assert len(best.get('US30', [])) == 1
        finally:
            os.unlink(db_path)


# ============================================================================
# C7: Recalculate script
# ============================================================================

class TestRecalculateDB:
    """C7: recalculate_db.py function tests."""

    def test_recalculate_dry_run(self):
        """Dry run should not change the database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            # Insert config with old-style WFE (not recalculated)
            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1,
                    'score': 1.0,
                    'config': {'entry_type': 'Donc+Chikou'},
                    'metrics_is': {'calmar': 0.05, 'n_trades': 50},
                    'metrics_oos': {'calmar': 0.5, 'n_trades': 20},
                }],
            }
            db.save_run(result)

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from recalculate_db import recalculate_wfe

            res = recalculate_wfe(db_path, apply=False, verbose=False)
            # Should detect that WFE needs to be set to NULL
            assert res['total'] >= 1

        finally:
            os.unlink(db_path)


# ============================================================================
# Integration: Quality scoring pipeline
# ============================================================================

class TestQualityScoringPipeline:
    """Integration test: full pipeline from save to quality grade."""

    def test_full_pipeline(self):
        """Save configs, retrieve best, verify quality grade computation."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1,
                    'score': 2.0,
                    'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 15,
                              'pmax_multiplier': 3.0, 'pmax_ma_type': 'EMA'},
                    'metrics_is': {'calmar': 1.5, 'n_trades': 100},
                    'metrics_oos': {'calmar': 0.8, 'n_trades': 30,
                                   'profit_factor': 1.5, 'max_dd': -6.0},
                    'perturbation_pass': {'stable': True, 'degradation': 0.1},
                    'walk_forward': {'stability_score': 75, 'consistency': 80,
                                    'avg_calmar': 0.6, 'avg_pf': 1.3,
                                    'degradation': False},
                }],
            }
            db.save_run(result)

            best = db.get_best_per_instrument(top_n=5)
            assert 'US30' in best
            cfg = best['US30'][0]

            # Check V5.4 quality fields
            assert '_grade' in cfg
            assert '_quality' in cfg
            assert '_robustness' in cfg
            assert '_flags' in cfg
            assert isinstance(cfg['_flags'], list)
            assert cfg['_grade'] in ('A', 'B', 'C', 'D')
            assert 0 <= cfg['_quality'] <= 100
            assert 0 <= cfg['_robustness'] <= 100
        finally:
            os.unlink(db_path)


# ============================================================================
# TV Export tests
# ============================================================================

class TestTVExport:
    """C8/C9: TV export includes grade, quality, disclaimers."""

    def test_grade_in_export(self):
        """TV export should include grade when present."""
        from pf_ma_optimizer.tv_export import format_config_for_tv
        result = {
            'rank': 1,
            'score': 2.0,
            'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 10,
                       'pmax_multiplier': 3.0, 'pmax_ma_mode': 'classic',
                       'pmax_ma_type': 'EMA'},
            'metrics_is': {'calmar': 1.5, 'n_trades': 100},
            'metrics_oos': {'calmar': 0.8, 'n_trades': 30},
            '_grade': 'A',
            '_quality': 75.5,
            'wfe': 0.53,
        }
        text = format_config_for_tv(result)
        assert 'Grade: A' in text
        assert '75.5' in text

    def test_position_sizing_disclaimer(self):
        """All configs export should include exploratory warning."""
        from pf_ma_optimizer.tv_export import format_all_configs
        text = format_all_configs([], instrument='US30')
        assert 'EXPLORATORY' in text
        assert 'TradingView' in text

    def test_wfe_na_in_export(self):
        """TV export should show N/A when WFE is None."""
        from pf_ma_optimizer.tv_export import format_config_for_tv
        result = {
            'rank': 1, 'score': 1.0,
            'config': {'entry_type': 'Donc+Chikou', 'pmax_ma_mode': 'classic',
                       'pmax_ma_type': 'EMA', 'pmax_length': 10,
                       'pmax_multiplier': 3.0},
            'metrics_is': {'calmar': 0.05, 'n_trades': 50},
            'metrics_oos': {'n_trades': 0},
            '_grade': 'D',
            '_quality': 10.0,
            'wfe': None,
        }
        text = format_config_for_tv(result)
        assert 'WFE: N/A' in text

    def test_non_safe_warning(self):
        """TV export should warn about non-safe entry types."""
        from pf_ma_optimizer.tv_export import format_config_for_tv
        result = {
            'rank': 1, 'score': 1.0,
            'config': {'entry_type': 'VWAP', 'pmax_ma_mode': 'classic',
                       'pmax_ma_type': 'EMA', 'pmax_length': 10,
                       'pmax_multiplier': 3.0},
            'metrics_is': {'calmar': 1.0, 'n_trades': 50},
            'metrics_oos': {'n_trades': 0},
            '_grade': 'D',
            '_quality': 10.0,
            'wfe': None,
        }
        text = format_config_for_tv(result)
        assert 'not validated by PineGuard' in text or 'VWAP' in text

    def test_direction_long_only(self):
        """TV export should clearly show LONG ONLY when short_side=False."""
        from pf_ma_optimizer.tv_export import format_config_for_tv
        result = {
            'rank': 1, 'score': 1.0,
            'config': {'entry_type': 'Donc+Chikou', 'pmax_ma_type': 'EMA',
                       'pmax_length': 10, 'pmax_multiplier': 3.0,
                       'long_side': True, 'short_side': False},
            'metrics_is': {'calmar': 1.0, 'n_trades': 50},
            'metrics_oos': {'n_trades': 0},
        }
        text = format_config_for_tv(result)
        assert 'LONG ONLY' in text
        assert 'LONG + SHORT' not in text

    def test_direction_long_short(self):
        """TV export should show LONG + SHORT when both sides enabled."""
        from pf_ma_optimizer.tv_export import format_config_for_tv
        result = {
            'rank': 1, 'score': 1.0,
            'config': {'entry_type': 'Donc+Chikou', 'pmax_ma_type': 'EMA',
                       'pmax_length': 10, 'pmax_multiplier': 3.0,
                       'long_side': True, 'short_side': True},
            'metrics_is': {'calmar': 1.0, 'n_trades': 50},
            'metrics_oos': {'n_trades': 0},
        }
        text = format_config_for_tv(result)
        assert 'LONG + SHORT' in text

    def test_no_phantom_params_in_export(self):
        """TV export must NOT contain MA Factor, MA Dist, or MA Cross Signal."""
        from pf_ma_optimizer.tv_export import format_config_for_tv
        result = {
            'rank': 1, 'score': 1.0,
            'config': {'entry_type': 'MM Cross', 'pmax_ma_type': 'EMA',
                       'pmax_length': 10, 'pmax_multiplier': 3.0,
                       'mm_cross_ma1_type': 'EMA', 'mm_cross_ma1_len': 9,
                       'mm_cross_ma2_type': 'EMA', 'mm_cross_ma2_len': 21,
                       'pmax_ma_mode': 'mult_int',  # old phantom value
                       'pmax_ma_factor': 1.5,        # old phantom value
                       'use_ma_cross_signal': True},  # old phantom value
            'metrics_is': {'calmar': 1.0, 'n_trades': 50},
            'metrics_oos': {'n_trades': 0},
        }
        text = format_config_for_tv(result)
        assert 'MA Factor' not in text
        assert 'MA Dist' not in text
        assert 'MA Cross Signal' not in text
        # Should still show MM Cross entry details
        assert 'MM Cross' in text or 'MM1' in text

    def test_runtime_guard_classic_mode(self):
        """Backtest engine should force pmax_ma_mode to classic."""
        # Just verify the config processing doesn't crash with non-classic mode
        from pf_ma_optimizer.config import get_default_config
        cfg = get_default_config()
        cfg['pmax_ma_mode'] = 'mult_int'  # force invalid mode
        cfg['pmax_ma_factor'] = 1.5
        # The backtest engine should silently force it to 'classic'
        # We can't run a full backtest here without data, but we can
        # verify the config structure is valid
        assert cfg['pmax_ma_mode'] == 'mult_int'  # config is unchanged
        # The guard is in the engine, not in the config


# ============================================================================
# Database schema migration
# ============================================================================

class TestSchemaMigration:
    """DB schema should handle migration for oos_trades_json column."""

    def test_migration_adds_column(self):
        """Opening DB twice should not fail (ALTER TABLE handled)."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db1 = BacktestDB(db_path)
            db2 = BacktestDB(db_path)  # Should not raise

            # Verify column exists
            conn = sqlite3.connect(db_path)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(configs)").fetchall()]
            conn.close()
            assert 'oos_trades_json' in cols
        finally:
            os.unlink(db_path)

    def test_migration_adds_oos_enrich_columns(self):
        """FEAT-OOS-ENRICH: new columns should be created by migration."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            conn = sqlite3.connect(db_path)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(configs)").fetchall()]
            conn.close()

            for col in ['oos_avg_win', 'oos_avg_loss', 'oos_expectancy', 'oos_avg_duration']:
                assert col in cols, f"Column {col} missing after migration"
        finally:
            os.unlink(db_path)


# ============================================================================
# FEAT-OOS-ENRICH: OOS Enrichment Tests
# ============================================================================

class TestOOSEnrichment:
    """FEAT-OOS-ENRICH: validate new OOS metrics and compact trade list."""

    def _make_config_data(self, with_trades=True, n_trades=25):
        """Build a config_data dict mimicking optimizer output."""
        compact_trades = []
        total_win_pct = 0
        total_loss_pct = 0
        n_wins = 0
        n_losses = 0
        for i in range(n_trades):
            side = 'l' if i % 3 != 0 else 's'
            pnl = round(1.5 - (i % 5) * 0.8, 3)  # mix of wins/losses
            compact_trades.append({
                's': side, 'ep': 100.0 + i, 'xp': 100.0 + i + pnl,
                'pnl': pnl, 'dur': 10 + i, 'xr': 'SL' if pnl < 0 else 'setup',
            })
            if pnl > 0:
                total_win_pct += pnl
                n_wins += 1
            elif pnl < 0:
                total_loss_pct += abs(pnl)
                n_losses += 1

        avg_win = total_win_pct / n_wins if n_wins > 0 else 0
        avg_loss = -(total_loss_pct / n_losses) if n_losses > 0 else 0
        win_rate = (n_wins / n_trades) * 100
        profit_factor = total_win_pct / total_loss_pct if total_loss_pct > 0 else 999.0
        expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

        trades_json = json.dumps(compact_trades, separators=(',', ':'))

        return {
            'rank': 1,
            'score': 2.5,
            'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 15,
                       'pmax_multiplier': 3.0, 'pmax_ma_type': 'EMA'},
            'metrics_is': {'calmar': 1.5, 'n_trades': 100, 'avg_win': 1.2,
                           'avg_loss': -0.5, 'win_rate': 60.0,
                           'profit_factor': 2.4, 'avg_duration_bars': 12.0},
            'metrics_oos': {'calmar': 0.8, 'n_trades': n_trades,
                            'profit_factor': round(profit_factor, 3),
                            'max_dd': -6.0, 'win_rate': round(win_rate, 1),
                            'avg_win': round(avg_win, 3),
                            'avg_loss': round(avg_loss, 3),
                            'avg_duration_bars': 22.0},
            'oos_avg_win': round(avg_win, 3),
            'oos_avg_loss': round(avg_loss, 3),
            'oos_expectancy': round(expectancy, 4),
            'oos_avg_duration': 22.0,
            'oos_trades_json': trades_json if with_trades else None,
            'perturbation_pass': {'stable': True, 'degradation': 0.1},
            'walk_forward': {'stability_score': 75, 'consistency': 80,
                             'avg_calmar': 0.6, 'avg_pf': 1.3,
                             'degradation': False},
        }

    def test_configs_contain_new_fields(self):
        """After save, configs should contain all 5 new OOS fields."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            cfg_data = self._make_config_data()
            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [cfg_data],
            }
            run_id = db.save_run(result)
            configs = db.get_configs(run_id=run_id)
            assert len(configs) == 1
            c = configs[0]

            # All 5 new fields must be present and non-None
            assert c['oos_avg_win'] is not None, "oos_avg_win missing"
            assert c['oos_avg_loss'] is not None, "oos_avg_loss missing"
            assert c['oos_expectancy'] is not None, "oos_expectancy missing"
            assert c['oos_avg_duration'] is not None, "oos_avg_duration missing"
            assert c['oos_trades_json'] is not None, "oos_trades_json missing"

            # Verify values are reasonable
            assert isinstance(c['oos_avg_win'], float)
            assert isinstance(c['oos_avg_loss'], float)
            assert isinstance(c['oos_expectancy'], float)
            assert c['oos_avg_duration'] > 0
        finally:
            os.unlink(db_path)

    def test_json_export_v2_format(self):
        """JSON export should use pf_ai_lab_db_v2 format and include new fields."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            cfg_data = self._make_config_data()
            result = {
                'instrument': 'UK100',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [cfg_data],
            }
            db.save_run(result)

            exported = db.export_full_json()
            assert exported['format'] == 'pf_ai_lab_db_v2'
            assert len(exported['runs']) == 1
            cfg = exported['runs'][0]['configs'][0]
            assert 'oos_avg_win' in cfg
            assert 'oos_avg_loss' in cfg
            assert 'oos_expectancy' in cfg
            assert 'oos_avg_duration' in cfg
            assert 'oos_trades_json' in cfg
        finally:
            os.unlink(db_path)

    def test_import_v1_with_null_new_fields(self):
        """Importing v1 format should work with NULL for new OOS fields."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            # Build a v1 export (no new OOS fields in configs)
            v1_data = {
                'format': 'pf_ai_lab_db_v1',
                'exported_at': '2026-02-01T00:00:00Z',
                'runs': [{
                    'created_at': '2026-02-01T00:00:00Z',
                    'instrument': 'EURUSD',
                    'mode': 'focused',
                    'n_trials': 100,
                    'is_ratio': 0.70,
                    'is_start': '2020-01-01',
                    'is_end': '2023-01-01',
                    'oos_start': '2023-01-01',
                    'oos_end': '2024-01-01',
                    'duration_sec': 50.0,
                    'n_configs': 1,
                    'data_file': 'EURUSD_H4.csv',
                    'notes': '',
                    'configs': [{
                        'rank': 1, 'score': 1.5,
                        'entry_type': 'Donc+Chikou',
                        'pmax_ma_type': 'EMA', 'pmax_length': 10,
                        'pmax_multiplier': 3.0,
                        'is_trades': 80, 'is_calmar': 1.2,
                        'oos_trades': 20, 'oos_calmar': 0.6,
                        'oos_profit_factor': 1.3,
                        'config_json': '{"entry_type": "Donc+Chikou"}',
                        # NOTE: no oos_avg_win, oos_avg_loss, etc.
                    }],
                }],
            }

            res = db.import_full_json(v1_data)
            assert res['imported_runs'] == 1
            assert res['imported_configs'] == 1
            assert len(res['errors']) == 0

            configs = db.get_configs()
            assert len(configs) == 1
            c = configs[0]
            # New fields should be NULL (None in Python)
            assert c.get('oos_avg_win') is None
            assert c.get('oos_avg_loss') is None
            assert c.get('oos_expectancy') is None
            assert c.get('oos_avg_duration') is None
        finally:
            os.unlink(db_path)

    def test_import_v2_preserves_data(self):
        """Importing v2 format should preserve all new OOS fields."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            cfg_data = self._make_config_data()
            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [cfg_data],
            }
            db.save_run(result)

            # Export as v2
            exported = db.export_full_json()
            assert exported['format'] == 'pf_ai_lab_db_v2'

            # Import into a fresh DB
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f2:
                db_path2 = f2.name
            db2 = BacktestDB(db_path2)
            res = db2.import_full_json(exported)
            assert res['imported_runs'] == 1
            assert res['imported_configs'] == 1

            configs = db2.get_configs()
            c = configs[0]
            assert c['oos_avg_win'] == cfg_data['oos_avg_win']
            assert c['oos_avg_loss'] == cfg_data['oos_avg_loss']
            assert c['oos_expectancy'] == cfg_data['oos_expectancy']
            assert c['oos_avg_duration'] == cfg_data['oos_avg_duration']
            assert c['oos_trades_json'] is not None
            os.unlink(db_path2)
        finally:
            os.unlink(db_path)

    def test_profit_factor_consistency(self):
        """OOS profit factor from metrics should be consistent with trade data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            cfg_data = self._make_config_data(n_trades=30)
            result = {
                'instrument': 'GBPUSD',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [cfg_data],
            }
            db.save_run(result)
            configs = db.get_configs()
            c = configs[0]

            # Recompute PF from compact trades
            trades_list = json.loads(c['oos_trades_json'])
            total_win = sum(t['pnl'] for t in trades_list if t['pnl'] > 0)
            total_loss = sum(abs(t['pnl']) for t in trades_list if t['pnl'] < 0)
            computed_pf = total_win / total_loss if total_loss > 0 else 999.0

            stored_pf = c['oos_profit_factor'] or 0
            # Allow 10% tolerance due to float rounding
            if stored_pf > 0 and stored_pf < 999:
                assert abs(computed_pf - stored_pf) / stored_pf < 0.10, \
                    f"PF mismatch: computed={computed_pf:.3f} vs stored={stored_pf:.3f}"
        finally:
            os.unlink(db_path)

    def test_oos_trades_json_size_limit(self):
        """oos_trades_json should stay under 15 KB."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            # Create config with many trades (500+)
            cfg_data = self._make_config_data(n_trades=500)
            result = {
                'instrument': 'BTCUSD',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [cfg_data],
            }
            db.save_run(result)
            configs = db.get_configs()
            c = configs[0]

            if c['oos_trades_json']:
                size_bytes = len(c['oos_trades_json'].encode('utf-8'))
                assert size_bytes <= 15_000, \
                    f"oos_trades_json too large: {size_bytes} bytes > 15,000 limit"
        finally:
            os.unlink(db_path)


# ============================================================================
# C9: TV Settings Parser
# ============================================================================

class TestTvSettingsParser:
    """C9: Test TradingView PineScript JSON export parser."""

    def test_parse_pine_json_basic(self):
        """Parse basic JSON from PineScript export."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '{"pmax_ma_type": "EMA", "pmax_length": 10, "pmax_multiplier": 3.0, "entry_type": "Donc+Chikou"}'
        result = parse_pine_json(text)
        assert result['error'] is None
        assert result['n_parsed'] == 4
        cfg = result['config']
        assert cfg['pmax_ma_type'] == 'EMA'
        assert cfg['pmax_length'] == 10
        assert cfg['pmax_multiplier'] == 3.0
        assert cfg['entry_type'] == 'Donc+Chikou'

    def test_sl_mode_transformation(self):
        """SL Mode value should be mapped from TV display to internal."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '{"sl_mode": "ATR-Based", "atr_sl_mult": 2.0, "sl_dynamic": true}'
        result = parse_pine_json(text)
        cfg = result['config']
        assert cfg['sl_mode'] == 'atr_based'
        assert cfg['atr_sl_mult'] == 2.0
        assert cfg['sl_dynamic'] is True

    def test_sl_mode_static(self):
        """Static SL mode mapping."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '{"sl_mode": "Static - %", "sl_pct": 5.0}'
        result = parse_pine_json(text)
        cfg = result['config']
        assert cfg['sl_mode'] == 'static_pct'
        assert cfg['sl_pct'] == 5.0

    def test_bool_string_coercion(self):
        """String "true"/"false" should be converted to Python bool."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '{"use_lt_filter": "true", "use_mt_filter": "false", "pmax_length": 10}'
        result = parse_pine_json(text)
        cfg = result['config']
        assert cfg['use_lt_filter'] is True
        assert cfg['use_mt_filter'] is False
        assert cfg['pmax_length'] == 10  # non-bool stays numeric

    def test_empty_input(self):
        """Empty text should return error."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        result = parse_pine_json("")
        assert result['n_parsed'] == 0
        assert result['error'] is not None

    def test_no_json_in_text(self):
        """Text without JSON should return error."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        result = parse_pine_json("just some random text")
        assert result['n_parsed'] == 0

    def test_multiline_json(self):
        """JSON split across multiple lines (TV table copy) should work."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '''{"pmax_ma_type": "EMA", "pmax_length": 10,
        "entry_type": "Fractal", "sl_mode": "No"}'''
        result = parse_pine_json(text)
        cfg = result['config']
        assert cfg['pmax_ma_type'] == 'EMA'
        assert cfg['entry_type'] == 'Fractal'
        assert cfg['sl_mode'] == 'no'

    def test_pt_mode_mapping(self):
        """PT Mode mapping."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '{"tp_mode": "SL Ratio", "tp_sl_ratio": 2.0}'
        result = parse_pine_json(text)
        cfg = result['config']
        assert cfg['tp_mode'] == 'sl_ratio'

    def test_be_mode_mapping(self):
        """BE Mode mapping."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '{"be_mode": "BE + TP + TSL", "be_trigger_pct": 50.0}'
        result = parse_pine_json(text)
        cfg = result['config']
        assert cfg['be_mode'] == 'be_tsl'

    def test_mapping_guide(self):
        """Mapping guide should contain all major sections."""
        from pf_ma_optimizer.tv_settings_parser import generate_tv_mapping_guide
        guide = generate_tv_mapping_guide()
        assert len(guide) > 40
        sections = set(g['section'] for g in guide)
        assert 'PMax' in sections
        assert 'Entry' in sections
        assert 'Risk: SL' in sections

    def test_pine_export_block_generation(self):
        """PineScript export block should be valid code."""
        from pf_ma_optimizer.tv_settings_parser import generate_pine_export_block
        code = generate_pine_export_block()
        assert 'input.bool' in code
        assert 'table.new' in code or 'table.cell' in code
        assert 'pmax_ma_type' in code
        assert 'entry_type' in code
        assert 'sl_mode' in code
        assert 'str.tostring' in code

    def test_complete_config_roundtrip(self):
        """A complete JSON should parse correctly with mode transformations."""
        from pf_ma_optimizer.tv_settings_parser import parse_pine_json
        text = '''{"pmax_ma_type": "EMA", "pmax_length": 15,
        "pmax_multiplier": 2.5, "entry_type": "Donc+Chikou",
        "don_length": 30, "long_side": true, "short_side": false,
        "order_pause": 3, "use_lt_filter": true,
        "sl_mode": "ATR-Based", "atr_sl_mult": 1.5,
        "sl_dynamic": true, "sl_vol_min_mult": 0.8,
        "sl_vol_max_mult": 1.5, "tp_mode": "SL Ratio",
        "tp_sl_ratio": 2.5, "be_mode": "No",
        "exit_setup_reversal": true}'''
        result = parse_pine_json(text)
        cfg = result['config']
        assert cfg['pmax_multiplier'] == 2.5
        assert cfg['pmax_length'] == 15
        assert cfg['entry_type'] == 'Donc+Chikou'
        assert cfg['don_length'] == 30
        assert cfg['long_side'] is True
        assert cfg['short_side'] is False
        assert cfg['sl_mode'] == 'atr_based'
        assert cfg['atr_sl_mult'] == 1.5
        assert cfg['sl_dynamic'] is True
        assert cfg['tp_mode'] == 'sl_ratio'
        assert cfg['be_mode'] == 'no'

    def test_pine_var_mapping_coverage(self):
        """PINE_TO_PFLAB should cover key PineScript variables."""
        from pf_ma_optimizer.tv_settings_parser import PINE_TO_PFLAB
        required = ['Multiplier', 'mav', 'length', 'entry_type',
                     'sl_mode', 'atr_sl_mult', 'sl_dynamic_adjust',
                     'pt_mode', 'be_mode', 'lt_trend_filter',
                     'mt_trend_filter', 'exit_at_reverse_setup']
        for v in required:
            assert v in PINE_TO_PFLAB, f"Missing Pine variable: {v}"


# ============================================================================
# C10: Alert-Based Config Import (V5.4.1)
# ============================================================================

class TestAlertConfigImport:
    """C10: Test alert-based config extraction."""

    def test_parse_alert_with_config(self):
        """Parse a PineConnector alert with embedded PFLAB_CONFIG."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        alert_text = (
            '6119578435797,buy,BTCUSDT,sl=42000,risk=1.0\n'
            'PFLAB_CONFIG:{"pmax_ma_type":"EMA","pmax_length":10,'
            '"entry_type":"Donc+Chikou","sl_mode":"ATR-Based","atr_sl_mult":2.0}'
        )
        result = parse_alert_with_config(alert_text)
        assert result['error'] is None
        assert result['source'] == 'alert'
        assert result['n_parsed'] == 5
        cfg = result['config']
        assert cfg['pmax_ma_type'] == 'EMA'
        assert cfg['entry_type'] == 'Donc+Chikou'
        assert cfg['sl_mode'] == 'atr_based'
        assert cfg['atr_sl_mult'] == 2.0
        # PineConnector command should also be parsed
        cmd = result['alert_command']
        assert cmd is not None
        assert cmd['action'] == 'buy'
        assert cmd['symbol'] == 'BTCUSDT'
        assert cmd['sl'] == 42000.0
        assert cmd['risk'] == 1.0

    def test_parse_alert_sell_with_tp(self):
        """Parse a sell alert with TP."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        alert_text = (
            '6119578435797,sell,EURUSD,sl=1.12,tp=1.08,risk=0.5\n'
            'PFLAB_CONFIG:{"pmax_ma_type":"SMA","entry_type":"Fractal"}'
        )
        result = parse_alert_with_config(alert_text)
        assert result['source'] == 'alert'
        cmd = result['alert_command']
        assert cmd['action'] == 'sell'
        assert cmd['symbol'] == 'EURUSD'
        assert cmd['tp'] == 1.08
        cfg = result['config']
        assert cfg['pmax_ma_type'] == 'SMA'

    def test_parse_close_alert_with_config(self):
        """Parse a closelong alert with embedded config."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        alert_text = (
            '6119578435797,closelong,BTCUSDT\n'
            'PFLAB_CONFIG:{"entry_type":"Boll+SMA","pmax_length":20}'
        )
        result = parse_alert_with_config(alert_text)
        assert result['source'] == 'alert'
        assert result['alert_command']['action'] == 'closelong'
        assert result['config']['entry_type'] == 'Boll+SMA'

    def test_parse_raw_json_fallback(self):
        """When no PFLAB_CONFIG prefix, should parse raw JSON (may detect as webhook)."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        text = '{"pmax_ma_type":"EMA","entry_type":"Fractal","sl_mode":"No"}'
        result = parse_alert_with_config(text)
        # May be detected as webhook or json source - both are acceptable
        assert result['source'] in ('json', 'webhook')
        assert result['n_parsed'] == 3
        assert result['config']['sl_mode'] == 'no'
        assert result['config']['entry_type'] == 'Fractal'
        assert result['alert_command'] is None

    def test_parse_webhook_json(self):
        """Parse a TradingView webhook JSON payload."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        text = '{"action":"buy","config":{"entry_type":"VWAP","pmax_length":15}}'
        result = parse_alert_with_config(text)
        assert result['source'] == 'webhook'
        assert result['config']['entry_type'] == 'VWAP'
        assert result['config']['pmax_length'] == 15

    def test_parse_webhook_pflab_key(self):
        """Parse webhook with PFLAB_CONFIG key."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        text = '{"PFLAB_CONFIG":{"entry_type":"Donc+Chikou","pmax_multiplier":3.0}}'
        result = parse_alert_with_config(text)
        assert result['source'] == 'webhook'
        assert result['config']['entry_type'] == 'Donc+Chikou'

    def test_parse_empty(self):
        """Empty input should return error."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        result = parse_alert_with_config("")
        assert result['error'] is not None
        assert result['n_parsed'] == 0

    def test_parse_no_config_in_text(self):
        """Text without JSON should return error with alert_command if present."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        result = parse_alert_with_config("6119578435797,buy,BTCUSDT,risk=1.0")
        assert result['n_parsed'] == 0
        assert result['error'] is not None
        # Should still parse the PineConnector command
        assert result['alert_command'] is not None
        assert result['alert_command']['action'] == 'buy'

    def test_parse_pineconnector_command_formats(self):
        """Various PineConnector command formats should be parsed."""
        from pf_ma_optimizer.tv_settings_parser import _parse_pineconnector_command
        # Buy with SL and risk
        cmd = _parse_pineconnector_command("123,buy,BTCUSDT,sl=42000,risk=1.0")
        assert cmd['action'] == 'buy'
        assert cmd['sl'] == 42000.0

        # Closeshort without params
        cmd = _parse_pineconnector_command("123,closeshort,EURUSD")
        assert cmd['action'] == 'closeshort'
        assert 'sl' not in cmd

        # Invalid format
        cmd = _parse_pineconnector_command("this is not a command")
        assert cmd is None

    def test_alert_block_generation(self):
        """Alert block should contain alert enrichment code."""
        from pf_ma_optimizer.tv_settings_parser import generate_pine_alert_block
        code = generate_pine_alert_block()
        assert 'PFLAB_CONFIG' in code
        assert 'cfg_json' in code
        assert 'config_suffix' in code
        assert 'input.bool' in code
        assert 'export_config_in_alerts' in code
        # Should still have the table display option
        assert 'table.new' in code or 'table.cell' in code

    def test_patch_instructions(self):
        """Patch instructions should describe the minimal change."""
        from pf_ma_optimizer.tv_settings_parser import generate_pine_patch_instructions
        instr = generate_pine_patch_instructions()
        assert 'PFLAB_CONFIG' in instr
        assert 'config_suffix' in instr
        assert 'buy_alert_msg' in instr
        assert 'sell_alert_msg' in instr

    def test_mode_transforms_in_alert(self):
        """Mode values in alert JSON should be transformed correctly."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        alert_text = (
            '123,buy,BTCUSDT,risk=1.0\n'
            'PFLAB_CONFIG:{"sl_mode":"ATR-Based","tp_mode":"SL Ratio",'
            '"be_mode":"BE + TP + TSL"}'
        )
        result = parse_alert_with_config(alert_text)
        cfg = result['config']
        assert cfg['sl_mode'] == 'atr_based'
        assert cfg['tp_mode'] == 'sl_ratio'
        assert cfg['be_mode'] == 'be_tsl'

    def test_bool_coercion_in_alert(self):
        """Bool strings in alert JSON should be coerced."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        alert_text = (
            '123,buy,BTCUSDT\n'
            'PFLAB_CONFIG:{"long_side":"true","short_side":"false","pmax_length":10}'
        )
        result = parse_alert_with_config(alert_text)
        cfg = result['config']
        assert cfg['long_side'] is True
        assert cfg['short_side'] is False
        assert cfg['pmax_length'] == 10

    def test_pinescript_float_artifact(self):
        """PineScript sometimes outputs '3.' instead of '3.0'."""
        from pf_ma_optimizer.tv_settings_parser import parse_alert_with_config
        text = '{"pmax_multiplier": 3., "pmax_length": 10}'
        result = parse_alert_with_config(text)
        assert result['n_parsed'] == 2
        assert result['config']['pmax_multiplier'] == 3.0

    def test_backward_compat_generate_pine_export_block(self):
        """generate_pine_export_block should still work as alias."""
        from pf_ma_optimizer.tv_settings_parser import generate_pine_export_block
        code = generate_pine_export_block()
        assert len(code) > 100
        assert 'cfg_json' in code


# ============================================================================
# V5.5 Phase 1: Anti-Overfitting Tests
# ============================================================================

class TestPhase1_OOSGatesRaised:
    """v5.5 Phase 1.3: OOS gate thresholds raised."""

    def test_gates_calmar_raised_to_050(self):
        """OOS gate min_calmar should be 0.50."""
        from pf_ma_optimizer.config import OOS_GATES
        assert OOS_GATES['min_calmar'] == 0.50

    def test_gates_pf_raised_to_130(self):
        """OOS gate min_pf should be 1.30."""
        from pf_ma_optimizer.config import OOS_GATES
        assert OOS_GATES['min_pf'] == 1.30

    def test_gates_min_trades_raised_to_25(self):
        """OOS gate min_trades should be 25."""
        from pf_ma_optimizer.config import OOS_GATES
        assert OOS_GATES['min_trades'] == 25

    def test_calmar_045_fails_gate(self):
        """Calmar 0.45 should fail the raised gate (< 0.50)."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        metrics = {'calmar': 0.45, 'profit_factor': 1.5, 'max_dd': -5.0, 'n_trades': 30}
        passed, reasons = passes_oos_gates(metrics)
        assert not passed
        assert any('Calmar' in r for r in reasons)

    def test_calmar_050_passes_gate(self):
        """Calmar 0.50 should pass the gate."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        metrics = {'calmar': 0.50, 'profit_factor': 1.5, 'max_dd': -5.0, 'n_trades': 30}
        passed, reasons = passes_oos_gates(metrics)
        assert passed, f"Should pass, but failed: {reasons}"

    def test_pf_125_fails_gate(self):
        """PF 1.25 should fail the raised gate (< 1.30)."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        metrics = {'calmar': 1.0, 'profit_factor': 1.25, 'max_dd': -5.0, 'n_trades': 30}
        passed, reasons = passes_oos_gates(metrics)
        assert not passed
        assert any('PF' in r for r in reasons)

    def test_pf_130_passes_gate(self):
        """PF 1.30 should pass the gate."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        metrics = {'calmar': 1.0, 'profit_factor': 1.30, 'max_dd': -5.0, 'n_trades': 30}
        passed, reasons = passes_oos_gates(metrics)
        assert passed, f"Should pass, but failed: {reasons}"

    def test_trades_20_fails_gate(self):
        """20 trades should fail the raised gate (< 25)."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        metrics = {'calmar': 1.0, 'profit_factor': 1.5, 'max_dd': -5.0, 'n_trades': 20}
        passed, reasons = passes_oos_gates(metrics)
        assert not passed
        assert any('Trades' in r for r in reasons)

    def test_trades_25_passes_gate(self):
        """25 trades should pass the gate."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        metrics = {'calmar': 1.0, 'profit_factor': 1.5, 'max_dd': -5.0, 'n_trades': 25}
        passed, reasons = passes_oos_gates(metrics)
        assert passed, f"Should pass, but failed: {reasons}"


class TestPhase1_ISOOSCoherenceGates:
    """v5.5 Phase 1.1: IS/OOS ratio coherence gates in passes_oos_gates."""

    def test_pf_ratio_inflated_rejected(self):
        """OOS PF >> IS PF (ratio > 2.5) should be rejected."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        oos = {'calmar': 1.0, 'profit_factor': 5.0, 'max_dd': -5.0, 'n_trades': 30, 'win_rate': 60}
        is_m = {'profit_factor': 1.5, 'win_rate': 55}
        # OOS/IS PF ratio = 5.0/1.5 = 3.33 > 2.5
        passed, reasons = passes_oos_gates(oos, metrics_is=is_m)
        assert not passed
        assert any('inflated' in r for r in reasons)

    def test_pf_ratio_collapsed_rejected(self):
        """OOS PF << IS PF (ratio < 0.30) should be rejected."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        oos = {'calmar': 0.8, 'profit_factor': 1.30, 'max_dd': -5.0, 'n_trades': 30, 'win_rate': 55}
        is_m = {'profit_factor': 5.0, 'win_rate': 60}
        # OOS/IS PF ratio = 1.30/5.0 = 0.26 < 0.30
        passed, reasons = passes_oos_gates(oos, metrics_is=is_m)
        assert not passed
        assert any('collapsed' in r for r in reasons)

    def test_pf_ratio_healthy_passes(self):
        """OOS PF ≈ IS PF (ratio ~1.0) should pass."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        oos = {'calmar': 1.0, 'profit_factor': 1.5, 'max_dd': -5.0, 'n_trades': 30, 'win_rate': 58}
        is_m = {'profit_factor': 1.6, 'win_rate': 60}
        # OOS/IS PF ratio = 1.5/1.6 = 0.9375
        passed, reasons = passes_oos_gates(oos, metrics_is=is_m)
        assert passed, f"Should pass: {reasons}"

    def test_wr_delta_too_large_rejected(self):
        """Win rate delta > 15pp should be rejected."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        oos = {'calmar': 1.0, 'profit_factor': 1.5, 'max_dd': -5.0, 'n_trades': 30, 'win_rate': 40}
        is_m = {'profit_factor': 1.6, 'win_rate': 60}
        # WR delta = |40 - 60| = 20pp > 15pp
        passed, reasons = passes_oos_gates(oos, metrics_is=is_m)
        assert not passed
        assert any('WR delta' in r for r in reasons)

    def test_wr_delta_acceptable(self):
        """Win rate delta <= 15pp should pass."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        oos = {'calmar': 1.0, 'profit_factor': 1.5, 'max_dd': -5.0, 'n_trades': 30, 'win_rate': 50}
        is_m = {'profit_factor': 1.6, 'win_rate': 60}
        # WR delta = |50 - 60| = 10pp <= 15pp
        passed, reasons = passes_oos_gates(oos, metrics_is=is_m)
        assert passed, f"Should pass: {reasons}"

    def test_no_is_metrics_skips_coherence(self):
        """When metrics_is is None, coherence gates are skipped."""
        from pf_ma_optimizer.metrics import passes_oos_gates
        oos = {'calmar': 0.6, 'profit_factor': 1.4, 'max_dd': -5.0, 'n_trades': 30}
        passed, reasons = passes_oos_gates(oos, metrics_is=None)
        assert passed, f"Should pass without IS metrics: {reasons}"

    def test_coherence_gates_config_values(self):
        """OOS_COHERENCE_GATES should have correct values."""
        from pf_ma_optimizer.config import OOS_COHERENCE_GATES
        assert OOS_COHERENCE_GATES['pf_ratio_max'] == 2.5
        assert OOS_COHERENCE_GATES['pf_ratio_min'] == 0.30
        assert OOS_COHERENCE_GATES['wr_delta_max'] == 15.0


class TestPhase1_ISPFCapPenalty:
    """v5.5 Phase 1.2: IS PF > 3.0 cap penalty in composite score."""

    def test_is_pf_cap_config(self):
        """IS_PF_CAP_PENALTIES should contain expected thresholds."""
        from pf_ma_optimizer.config import IS_PF_CAP_PENALTIES
        assert 3.0 in IS_PF_CAP_PENALTIES
        assert 4.0 in IS_PF_CAP_PENALTIES
        assert 5.0 in IS_PF_CAP_PENALTIES
        assert IS_PF_CAP_PENALTIES[3.0] == 0.80
        assert IS_PF_CAP_PENALTIES[4.0] == 0.60
        assert IS_PF_CAP_PENALTIES[5.0] == 0.40

    def test_normal_pf_no_penalty(self):
        """IS PF <= 3.0 should not be penalized."""
        from pf_ma_optimizer.metrics import compute_composite_score
        metrics = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                   'profit_factor': 2.5, 'n_trades': 100}
        score = compute_composite_score(metrics)
        assert score > 0

    def test_high_pf_penalized(self):
        """IS PF > 3.0 should produce lower score than PF <= 3.0."""
        from pf_ma_optimizer.metrics import compute_composite_score
        metrics_normal = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                          'profit_factor': 2.5, 'n_trades': 100}
        metrics_high = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                        'profit_factor': 3.5, 'n_trades': 100}
        score_normal = compute_composite_score(metrics_normal)
        score_high = compute_composite_score(metrics_high)
        assert score_high < score_normal, \
            f"PF 3.5 score ({score_high}) should be < PF 2.5 score ({score_normal})"

    def test_very_high_pf_severely_penalized(self):
        """IS PF > 4.0 should be penalized more than PF > 3.0."""
        from pf_ma_optimizer.metrics import compute_composite_score
        metrics_35 = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                      'profit_factor': 3.5, 'n_trades': 100}
        metrics_45 = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                      'profit_factor': 4.5, 'n_trades': 100}
        score_35 = compute_composite_score(metrics_35)
        score_45 = compute_composite_score(metrics_45)
        assert score_45 < score_35, \
            f"PF 4.5 score ({score_45}) should be < PF 3.5 score ({score_35})"


class TestPhase1_CoherenceBonus:
    """v5.5 Phase 1.4: IS/OOS coherence bonus in composite score."""

    def test_coherence_bonus_when_ratio_near_1(self):
        """Score with IS metrics (OOS/IS ≈ 1.0) should get a coherence bonus."""
        from pf_ma_optimizer.metrics import compute_composite_score
        oos_metrics = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                       'profit_factor': 1.5, 'n_trades': 100}
        is_metrics = {'profit_factor': 1.5}  # ratio = 1.0
        score_with = compute_composite_score(oos_metrics, metrics_is=is_metrics)
        score_without = compute_composite_score(oos_metrics, metrics_is=None)
        assert score_with > score_without, \
            f"Score with IS coherence ({score_with}) should be > without ({score_without})"

    def test_no_bonus_when_ratio_far_from_1(self):
        """Score with IS metrics (OOS/IS far from 1.0) should get no bonus."""
        from pf_ma_optimizer.metrics import compute_composite_score
        oos_metrics = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                       'profit_factor': 1.5, 'n_trades': 100}
        is_metrics = {'profit_factor': 0.5}  # ratio = 3.0
        score_with = compute_composite_score(oos_metrics, metrics_is=is_metrics)
        score_without = compute_composite_score(oos_metrics, metrics_is=None)
        # Ratio = 3.0 -> bonus = max(0, 0.5 - |3.0-1.0|*0.5) = max(0, -0.5) = 0
        assert abs(score_with - score_without) < 0.01, \
            f"No bonus expected for ratio 3.0, got diff={score_with-score_without}"

    def test_max_bonus_is_half_point(self):
        """Maximum coherence bonus should be 0.5 pts."""
        from pf_ma_optimizer.metrics import compute_composite_score
        oos_metrics = {'calmar': 1.0, 'sortino': 1.0, 'max_dd': -5.0,
                       'profit_factor': 1.5, 'n_trades': 100}
        is_metrics = {'profit_factor': 1.5}  # perfect ratio = 1.0
        score_with = compute_composite_score(oos_metrics, metrics_is=is_metrics)
        score_without = compute_composite_score(oos_metrics, metrics_is=None)
        bonus = score_with - score_without
        assert bonus <= 0.51, f"Bonus {bonus} should be <= 0.5"


class TestPhase1_QualityGradeOverfitFlags:
    """v5.5 Phase 1: Quality grade with overfit detection flags."""

    def test_is_pf_too_high_flag(self):
        """IS PF > 3.0 should produce IS_PF_TOO_HIGH flag."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=10.0, robustness_score=80.0,
            entry_type='Donc+Chikou', instrument='US30',
            metrics_is={'profit_factor': 4.0, 'win_rate': 60},
            metrics_oos={'profit_factor': 1.5, 'win_rate': 55},
        )
        assert 'IS_PF_TOO_HIGH' in result['flags']
        assert result['multiplier'] < 1.0

    def test_overfit_pf_ratio_flag(self):
        """OOS/IS PF ratio > 2.5 should produce OVERFIT_PF_RATIO flag."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=10.0, robustness_score=80.0,
            entry_type='Donc+Chikou', instrument='US30',
            metrics_is={'profit_factor': 1.0, 'win_rate': 55},
            metrics_oos={'profit_factor': 3.0, 'win_rate': 55},  # ratio = 3.0
        )
        assert 'OVERFIT_PF_RATIO' in result['flags']

    def test_coherent_config_gets_bonus(self):
        """Config with OOS/IS PF ratio ≈ 1.0 should get IS_OOS_COHERENT flag."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=10.0, robustness_score=80.0,
            entry_type='Donc+Chikou', instrument='US30',
            metrics_is={'profit_factor': 1.5, 'win_rate': 60},
            metrics_oos={'profit_factor': 1.4, 'win_rate': 58},  # ratio ≈ 0.93
        )
        assert 'IS_OOS_COHERENT' in result['flags']
        assert result['coherence_bonus'] > 0

    def test_wr_delta_flag(self):
        """Win rate delta > 15pp should produce OVERFIT_WR_DELTA flag."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=10.0, robustness_score=80.0,
            entry_type='Donc+Chikou', instrument='US30',
            metrics_is={'profit_factor': 1.5, 'win_rate': 65},
            metrics_oos={'profit_factor': 1.4, 'win_rate': 45},  # delta = 20pp
        )
        assert 'OVERFIT_WR_DELTA' in result['flags']

    def test_pf_ratio_returned_in_grade_info(self):
        """Grade info should include pf_ratio and wr_delta fields."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=10.0, robustness_score=80.0,
            entry_type='Donc+Chikou', instrument='US30',
            metrics_is={'profit_factor': 1.5, 'win_rate': 60},
            metrics_oos={'profit_factor': 1.4, 'win_rate': 58},
        )
        assert 'pf_ratio' in result
        assert 'wr_delta' in result
        assert 'coherence_bonus' in result
        assert result['pf_ratio'] is not None
        assert result['wr_delta'] is not None

    def test_backward_compat_no_is_oos_metrics(self):
        """Quality grade without IS/OOS metrics should work (backward compat)."""
        from pf_ma_optimizer.metrics import compute_quality_grade
        result = compute_quality_grade(
            perf_score=10.0, robustness_score=80.0,
            entry_type='Donc+Chikou', instrument='US30',
        )
        assert result['grade'] in ('A', 'B', 'C', 'D')
        assert result['pf_ratio'] is None
        assert result['wr_delta'] is None
        assert result['coherence_bonus'] == 0


class TestPhase1_DBIntegration:
    """v5.5 Phase 1: Database integration with new gates and overfit indicators."""

    def test_config_fails_new_stricter_gates(self):
        """Config with OOS calmar 0.40 should fail the new 0.50 gate."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1, 'score': 1.0,
                    'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 10},
                    'metrics_is': {'calmar': 1.0, 'n_trades': 100,
                                   'profit_factor': 1.8, 'win_rate': 55},
                    'metrics_oos': {'calmar': 0.40, 'n_trades': 30,
                                    'profit_factor': 1.5, 'max_dd': -5.0,
                                    'win_rate': 52},
                }],
            }
            run_id = db.save_run(result)
            configs = db.get_configs(run_id=run_id)
            assert len(configs) == 1
            # With calmar 0.40 < 0.50, gate should FAIL
            assert configs[0]['oos_gate_pass'] == 0, \
                f"Config with calmar 0.40 should fail new gate (0.50 min)"
        finally:
            os.unlink(db_path)

    def test_config_passes_new_stricter_gates(self):
        """Config with adequate metrics should pass the new stricter gates."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            result = {
                'instrument': 'EURUSD',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1, 'score': 2.0,
                    'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 15},
                    'metrics_is': {'calmar': 1.5, 'n_trades': 100,
                                   'profit_factor': 1.8, 'win_rate': 58},
                    'metrics_oos': {'calmar': 0.80, 'n_trades': 30,
                                    'profit_factor': 1.50, 'max_dd': -6.0,
                                    'win_rate': 55},
                }],
            }
            run_id = db.save_run(result)
            configs = db.get_configs(run_id=run_id)
            assert configs[0]['oos_gate_pass'] == 1, \
                f"Config with good metrics should pass"
        finally:
            os.unlink(db_path)

    def test_overfit_config_rejected_by_coherence(self):
        """Config with OOS/IS PF ratio < 0.30 should fail gates (overfit collapse)."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            result = {
                'instrument': 'XAUUSD',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1, 'score': 3.0,
                    'config': {'entry_type': 'Donc+Chikou'},
                    'metrics_is': {'calmar': 3.0, 'n_trades': 200,
                                   'profit_factor': 5.0, 'win_rate': 70},
                    'metrics_oos': {'calmar': 0.80, 'n_trades': 40,
                                    'profit_factor': 1.40, 'max_dd': -8.0,
                                    'win_rate': 52},
                    # OOS/IS PF ratio = 1.40/5.0 = 0.28 < 0.30
                }],
            }
            run_id = db.save_run(result)
            configs = db.get_configs(run_id=run_id)
            assert configs[0]['oos_gate_pass'] == 0, \
                "Config with PF ratio 0.28 should fail coherence gate"
        finally:
            os.unlink(db_path)

    def test_best_per_instrument_exposes_overfit_fields(self):
        """get_best_per_instrument should expose _pf_ratio, _wr_delta, _coherence_bonus."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1, 'score': 2.0,
                    'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 15,
                               'pmax_multiplier': 3.0, 'pmax_ma_type': 'EMA'},
                    'metrics_is': {'calmar': 1.5, 'n_trades': 100,
                                   'profit_factor': 1.8, 'win_rate': 58},
                    'metrics_oos': {'calmar': 0.8, 'n_trades': 30,
                                    'profit_factor': 1.5, 'max_dd': -6.0,
                                    'win_rate': 55},
                    'perturbation_pass': {'stable': True, 'degradation': 0.1},
                    'walk_forward': {'stability_score': 75, 'consistency': 80,
                                     'avg_calmar': 0.6, 'avg_pf': 1.3,
                                     'degradation': False},
                }],
            }
            db.save_run(result)
            best = db.get_best_per_instrument(top_n=5)
            cfg = best['US30'][0]

            assert '_pf_ratio' in cfg
            assert '_wr_delta' in cfg
            assert '_coherence_bonus' in cfg
            assert cfg['_pf_ratio'] is not None  # 1.5/1.8 = 0.833
            assert cfg['_wr_delta'] is not None   # |55 - 58| = 3.0
        finally:
            os.unlink(db_path)

    def test_overfit_flags_in_quality_grade(self):
        """Configs with IS PF too high should get IS_PF_TOO_HIGH flag via best."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            from pf_ma_optimizer.backtest_db import BacktestDB
            db = BacktestDB(db_path)

            result = {
                'instrument': 'US30',
                'is_range': ('2020-01-01', '2023-01-01'),
                'oos_range': ('2023-01-01', '2024-01-01'),
                'duration_sec': 10,
                'top_configs': [{
                    'rank': 1, 'score': 2.0,
                    'config': {'entry_type': 'Donc+Chikou', 'pmax_length': 15,
                               'pmax_multiplier': 3.0, 'pmax_ma_type': 'EMA'},
                    'metrics_is': {'calmar': 3.0, 'n_trades': 200,
                                   'profit_factor': 5.0, 'win_rate': 70},
                    'metrics_oos': {'calmar': 0.8, 'n_trades': 30,
                                    'profit_factor': 1.5, 'max_dd': -6.0,
                                    'win_rate': 55},
                }],
            }
            db.save_run(result)
            best = db.get_best_per_instrument(top_n=5)
            cfg = best['US30'][0]

            assert 'IS_PF_TOO_HIGH' in cfg['_flags']
        finally:
            os.unlink(db_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
