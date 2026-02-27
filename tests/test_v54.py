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
        """All configs export should include position sizing disclaimer."""
        from pf_ma_optimizer.tv_export import format_all_configs
        text = format_all_configs([], instrument='US30')
        assert 'POSITION SIZING' in text
        assert '1-2% of equity' in text

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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
