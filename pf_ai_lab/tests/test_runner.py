"""
Tests for PF AI Lab 5.0.4 runner (full backtest pipeline).
"""

import pytest
import pandas as pd
import numpy as np

from pineguard.config import PineGuardConfig
from pineguard.utils.data_loader import generate_sample_data


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data."""
    return generate_sample_data(n_bars=500, freq='1h', base_price=5000,
                                 volatility=0.001, seed=42)


@pytest.fixture
def default_config():
    return PineGuardConfig(
        instrument='US500', timeframe='1H',
        entry_type='Donc+Chikou',
        don_length=20, chikou_shift=26,
        pmax_length=10, pmax_multiplier=3.0,
        lt_filter_enabled=False,
        mt_filter_enabled=False,
    )


class TestRunBacktest:
    """Test the full backtest pipeline."""

    def test_run_backtest_returns_required_keys(self, sample_data, default_config):
        from pf_ai_lab.runner import run_backtest
        result = run_backtest(df=sample_data, config=default_config, verbose=False)

        assert 'trades' in result
        assert 'equity' in result
        assert 'metrics' in result
        assert 'signals' in result
        assert 'filters' in result
        assert 'pmax' in result
        assert 'config' in result
        assert 'validation' in result

    def test_run_backtest_metrics_complete(self, sample_data, default_config):
        from pf_ai_lab.runner import run_backtest
        result = run_backtest(df=sample_data, config=default_config, verbose=False)

        metrics = result['metrics']
        required_keys = [
            'total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio',
            'max_drawdown_pct', 'total_pnl', 'calmar_ratio', 'sortino_ratio',
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_run_backtest_sample_data_no_csv(self, default_config):
        from pf_ai_lab.runner import run_backtest
        result = run_backtest(config=default_config, n_sample_bars=300, verbose=False)
        assert 'error' not in result
        assert result['metrics']['total_trades'] >= 0

    def test_run_backtest_equity_length(self, sample_data, default_config):
        from pf_ai_lab.runner import run_backtest
        result = run_backtest(df=sample_data, config=default_config, verbose=False)
        assert len(result['equity']) == len(sample_data)

    def test_run_backtest_invalid_config(self):
        from pf_ai_lab.runner import run_backtest
        config = PineGuardConfig(pmax_length=-1)
        result = run_backtest(config=config, verbose=False)
        assert 'error' in result

    def test_run_backtest_all_entry_types(self, sample_data):
        """Test that all entry types can run without error."""
        from pf_ai_lab.runner import run_backtest

        for entry_type in ['Donc+Chikou', 'RSI Divergence', 'Boll+SMA', 'Fractal', 'MM Cross']:
            config = PineGuardConfig(
                entry_type=entry_type,
                lt_filter_enabled=False,
                mt_filter_enabled=False,
            )
            result = run_backtest(df=sample_data.copy(), config=config, verbose=False)
            assert 'error' not in result, f"Error for {entry_type}: {result.get('error')}"

    def test_run_backtest_with_filters(self, sample_data):
        """Test backtest with LT/MT filters enabled."""
        from pf_ai_lab.runner import run_backtest
        config = PineGuardConfig(
            entry_type='Donc+Chikou',
            lt_filter_enabled=True, lt_filter_tf='4H',
            mt_filter_enabled=True, mt_filter_tf='2H',
        )
        result = run_backtest(df=sample_data, config=config, verbose=False)
        assert 'error' not in result


class TestRunFullPipeline:
    """Test the full pipeline with audit."""

    def test_pipeline_with_audit(self, sample_data, default_config):
        from pf_ai_lab.runner import run_full_pipeline
        result = run_full_pipeline(
            df=sample_data, config=default_config,
            run_audit=True, verbose=False,
        )

        assert 'audit' in result
        assert 'confidence' in result['audit']
        assert 'divergence_summary' in result['audit']

    def test_pipeline_donc_chikou_confidence_100(self, sample_data):
        from pf_ai_lab.runner import run_full_pipeline
        config = PineGuardConfig(
            entry_type='Donc+Chikou',
            lt_filter_enabled=False, mt_filter_enabled=False,
        )
        result = run_full_pipeline(
            df=sample_data, config=config,
            run_audit=True, verbose=False,
        )
        assert result['audit']['confidence'] == 1.00


class TestEquityBugFix:
    """Test that the O(n^2) equity bug is fixed."""

    def test_equity_incremental(self, sample_data, default_config):
        """Equity should be computed incrementally, not O(n^2)."""
        from pf_ai_lab.runner import run_backtest
        import time

        # Small dataset should be very fast
        t0 = time.time()
        result = run_backtest(df=sample_data, config=default_config, verbose=False)
        duration = time.time() - t0

        # 500 bars should complete in under 2 seconds
        assert duration < 2.0, f"Backtest took {duration:.1f}s — O(n^2) bug?"
