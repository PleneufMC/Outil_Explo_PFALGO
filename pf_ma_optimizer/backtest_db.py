"""
PF AI Lab 5.0 — Backtest Database (SQLite)
Stores all optimization runs and their top configs with full metrics.

Tables:
  - runs:    one row per optimization launch (instrument, mode, date, etc.)
  - configs: one row per top-N config within a run (rank, score, metrics IS/OOS, WF, etc.)

Usage:
  from pf_ma_optimizer.backtest_db import BacktestDB
  db = BacktestDB()                       # default: data/backtests.db
  run_id = db.save_run(result_dict)       # after optimization
  rows   = db.query_configs(instrument='US30')
  csv    = db.export_csv()                # all configs as CSV string
"""

import os
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


def _safe_int(val, default=0):
    """Safely convert a value to int, handling None, NaN, Inf, numpy types."""
    if val is None:
        return default
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return default
        return int(fval)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(val, default=None):
    """Safely convert a value to float, replacing NaN/Inf with default."""
    if val is None:
        return default
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return default
        return fval
    except (TypeError, ValueError, OverflowError):
        return default


DB_FILENAME = 'backtests.db'


def _config_fingerprint_for_dedup(cfg: dict) -> str:
    """Compact hash for leaderboard dedup. Only trading-relevant params."""
    KEYS = [
        'entry_type', 'pmax_ma_type', 'pmax_length', 'pmax_multiplier',
        'pmax_ma_mode', 'pmax_ma_factor', 'pmax_ma_distance',
        'don_length', 'rsi_div_length', 'rsi_div_pivot', 'rsi_div_hidden',
        'mm_cross_ma1_type', 'mm_cross_ma1_len', 'mm_cross_ma2_type', 'mm_cross_ma2_len',
        'fractal_buffer', 'length_bb_ma', 'mult_bb_ma', 'length_bb_sma',
        'use_lt_filter', 'lt_multiplier',
        'use_mt_filter', 'use_rsi50_filter',
        'use_diff_ma_red', 'diff_ma_red_pct',
        'sl_mode', 'sl_pct', 'atr_sl_mult',
        'tp_mode', 'be_mode', 'long_side', 'short_side',
    ]
    parts = []
    for k in sorted(KEYS):
        v = cfg.get(k)
        if v is not None:
            if isinstance(v, float):
                v = round(v, 2)
            parts.append(f"{k}={v}")
    return '|'.join(parts)


class BacktestDB:
    """SQLite-backed backtest results database."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', DB_FILENAME)
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT    NOT NULL,
                    instrument      TEXT    NOT NULL,
                    mode            TEXT    NOT NULL DEFAULT 'focused',
                    n_trials        INTEGER NOT NULL DEFAULT 500,
                    is_ratio        REAL    NOT NULL DEFAULT 0.70,
                    is_start        TEXT,
                    is_end          TEXT,
                    oos_start       TEXT,
                    oos_end         TEXT,
                    duration_sec    REAL,
                    n_configs       INTEGER NOT NULL DEFAULT 0,
                    data_file       TEXT,
                    notes           TEXT
                );

                CREATE TABLE IF NOT EXISTS configs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    rank            INTEGER NOT NULL,
                    score           REAL,
                    -- Strategy params
                    entry_type      TEXT,
                    pmax_ma_type    TEXT,
                    pmax_length     INTEGER,
                    pmax_multiplier REAL,
                    -- LT Trend
                    use_lt_filter   INTEGER,
                    lt_tf           TEXT,
                    lt_length       INTEGER,
                    lt_ma_type      TEXT,
                    lt_multiplier   REAL,
                    -- MT Trend
                    use_mt_filter   INTEGER,
                    mt_tf           TEXT,
                    mt_length       INTEGER,
                    mt_ma_type      TEXT,
                    mt_multiplier   REAL,
                    -- Other filters
                    use_rsi50_filter    INTEGER,
                    use_diff_ma_red     INTEGER,
                    diff_ma_red_pct     REAL,
                    use_ma_dir_filter   INTEGER,
                    use_mavg_filter     INTEGER,
                    -- Entry-specific
                    don_length      INTEGER,
                    rsi_div_length  INTEGER,
                    rsi_div_pivot   INTEGER,
                    rsi_div_hidden  INTEGER,
                    fractal_buffer  REAL,
                    length_bb_ma    INTEGER,
                    mult_bb_ma      REAL,
                    length_bb_sma   INTEGER,
                    -- Risk
                    sl_mode         TEXT,
                    sl_pct          REAL,
                    order_pause     INTEGER,
                    long_side       INTEGER,
                    short_side      INTEGER,
                    -- IS Metrics
                    is_trades       INTEGER,
                    is_win_rate     REAL,
                    is_net_return   REAL,
                    is_max_dd       REAL,
                    is_calmar       REAL,
                    is_sharpe       REAL,
                    is_sortino      REAL,
                    is_profit_factor REAL,
                    is_avg_win      REAL,
                    is_avg_loss     REAL,
                    -- OOS Metrics
                    oos_trades      INTEGER,
                    oos_win_rate    REAL,
                    oos_net_return  REAL,
                    oos_max_dd      REAL,
                    oos_calmar      REAL,
                    oos_sharpe      REAL,
                    oos_sortino     REAL,
                    oos_profit_factor REAL,
                    -- OOS Gate
                    oos_gate_pass   INTEGER,
                    -- WFE
                    wfe             REAL,
                    -- Perturbation
                    perturbation_stable INTEGER,
                    perturbation_degradation REAL,
                    -- Walk-Forward
                    wf_stability    REAL,
                    wf_consistency  REAL,
                    wf_avg_calmar   REAL,
                    wf_avg_pf       REAL,
                    wf_degradation  INTEGER,
                    -- Full config JSON (for re-import)
                    config_json     TEXT,
                    tv_export       TEXT,
                    oos_trades_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_configs_run ON configs(run_id);
                CREATE INDEX IF NOT EXISTS idx_configs_instrument ON configs(run_id, rank);
                CREATE INDEX IF NOT EXISTS idx_runs_instrument ON runs(instrument);
                CREATE INDEX IF NOT EXISTS idx_runs_date ON runs(created_at);
            """)

            # V5.4: add oos_trades_json column if missing (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE configs ADD COLUMN oos_trades_json TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ──────────────────────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────────────────────

    def save_run(self, result: dict, mode: str = 'focused',
                 n_trials: int = 500, is_ratio: float = 0.70,
                 data_file: str = '', notes: str = '') -> int:
        """
        Save a complete optimization run (from /api/optimization_result data).
        Returns the run_id.
        """
        now = datetime.now(timezone.utc).isoformat()
        instrument = result.get('instrument', 'UNKNOWN')
        is_range = result.get('is_range', (None, None))
        oos_range = result.get('oos_range', (None, None))
        duration = result.get('duration_sec', 0)
        top_configs = result.get('top_configs', result.get('configs', []))

        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO runs (created_at, instrument, mode, n_trials, is_ratio,
                                  is_start, is_end, oos_start, oos_end,
                                  duration_sec, n_configs, data_file, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, instrument, mode, n_trials, is_ratio,
                  is_range[0] if is_range else None,
                  is_range[1] if is_range else None,
                  oos_range[0] if oos_range else None,
                  oos_range[1] if oos_range else None,
                  duration, len(top_configs), data_file, notes))
            run_id = cur.lastrowid

            for cfg_data in top_configs:
                self._insert_config(conn, run_id, cfg_data, instrument)

            conn.commit()

        return run_id

    def _insert_config(self, conn: sqlite3.Connection, run_id: int,
                        cfg_data: dict, instrument: str = 'UNKNOWN'):
        """Insert a single config row with adaptive OOS gate per instrument."""
        from .config import get_oos_dd_limit

        c = cfg_data.get('config', {})
        mis = cfg_data.get('metrics_is', {})
        moos = cfg_data.get('metrics_oos', {})
        perturb = cfg_data.get('perturbation_pass', {}) or {}
        wf = cfg_data.get('walk_forward', {}) or {}

        # OOS gate check — adaptive DD threshold per instrument asset class
        dd_limit = get_oos_dd_limit(instrument)
        oos_pass = (
            (moos.get('calmar', 0) or 0) >= 0.25 and
            (moos.get('profit_factor', 0) or 0) >= 1.2 and
            (moos.get('max_dd', 0) or 0) >= dd_limit and
            (moos.get('n_trades', 0) or 0) >= 15
        )

        # WFE — V5.4: floor + cap + None for degenerate cases
        is_calmar = _safe_float(mis.get('calmar', 0), 0) or 0
        oos_calmar = _safe_float(moos.get('calmar', 0), 0) or 0

        if is_calmar < 0.10:
            # IS Calmar too low → WFE not reliable (division instable)
            wfe = None
        elif oos_calmar <= 0:
            wfe = 0.0
        else:
            wfe = min(oos_calmar / is_calmar, 3.0)  # Cap at 300%

        wfe = _safe_float(wfe, None)  # None stored as NULL in SQLite

        # V5.4: OOS trades detail for audit/distribution analysis
        oos_trades_detail = cfg_data.get('oos_trades_detail')
        oos_trades_json = json.dumps(oos_trades_detail) if oos_trades_detail else None

        conn.execute("""
            INSERT INTO configs (
                run_id, rank, score,
                entry_type, pmax_ma_type, pmax_length, pmax_multiplier,
                use_lt_filter, lt_tf, lt_length, lt_ma_type, lt_multiplier,
                use_mt_filter, mt_tf, mt_length, mt_ma_type, mt_multiplier,
                use_rsi50_filter, use_diff_ma_red, diff_ma_red_pct,
                use_ma_dir_filter, use_mavg_filter,
                don_length, rsi_div_length, rsi_div_pivot, rsi_div_hidden,
                fractal_buffer, length_bb_ma, mult_bb_ma, length_bb_sma,
                sl_mode, sl_pct, order_pause, long_side, short_side,
                is_trades, is_win_rate, is_net_return, is_max_dd,
                is_calmar, is_sharpe, is_sortino, is_profit_factor,
                is_avg_win, is_avg_loss,
                oos_trades, oos_win_rate, oos_net_return, oos_max_dd,
                oos_calmar, oos_sharpe, oos_sortino, oos_profit_factor,
                oos_gate_pass, wfe,
                perturbation_stable, perturbation_degradation,
                wf_stability, wf_consistency, wf_avg_calmar, wf_avg_pf, wf_degradation,
                config_json, tv_export, oos_trades_json
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?
            )
        """, (
            run_id, cfg_data.get('rank', 0), cfg_data.get('score', 0),
            c.get('entry_type'), c.get('pmax_ma_type'), c.get('pmax_length'), c.get('pmax_multiplier'),
            int(c.get('use_lt_filter', False)), c.get('lt_tf'), c.get('lt_length'), c.get('lt_ma_type'), c.get('lt_multiplier'),
            int(c.get('use_mt_filter', False)), c.get('mt_tf'), c.get('mt_length'), c.get('mt_ma_type'), c.get('mt_multiplier'),
            int(c.get('use_rsi50_filter', False)), int(c.get('use_diff_ma_red', False)), c.get('diff_ma_red_pct'),
            int(c.get('use_ma_dir_filter', False)), int(c.get('use_mavg_filter', False)),
            c.get('don_length'), c.get('rsi_div_length'), c.get('rsi_div_pivot'),
            int(c.get('rsi_div_hidden', False)) if c.get('rsi_div_hidden') is not None else None,
            c.get('fractal_buffer'), c.get('length_bb_ma'), c.get('mult_bb_ma'), c.get('length_bb_sma'),
            c.get('sl_mode'), _safe_float(c.get('sl_pct')), c.get('order_pause'),
            _safe_int(c.get('long_side', True), 1), _safe_int(c.get('short_side', True), 1),
            _safe_int(mis.get('n_trades')), _safe_float(mis.get('win_rate')), _safe_float(mis.get('net_return')), _safe_float(mis.get('max_dd')),
            _safe_float(mis.get('calmar')), _safe_float(mis.get('sharpe')), _safe_float(mis.get('sortino')), _safe_float(mis.get('profit_factor')),
            _safe_float(mis.get('avg_win')), _safe_float(mis.get('avg_loss')),
            _safe_int(moos.get('n_trades')), _safe_float(moos.get('win_rate')), _safe_float(moos.get('net_return')), _safe_float(moos.get('max_dd')),
            _safe_float(moos.get('calmar')), _safe_float(moos.get('sharpe')), _safe_float(moos.get('sortino')), _safe_float(moos.get('profit_factor')),
            _safe_int(oos_pass), wfe,  # V5.4: None stored as NULL (not 0)
            _safe_int(perturb.get('stable', False)), _safe_float(perturb.get('degradation')),
            _safe_float(wf.get('stability_score')), _safe_float(wf.get('consistency')),
            _safe_float(wf.get('avg_calmar')), _safe_float(wf.get('avg_pf')), _safe_int(wf.get('degradation', False)),
            json.dumps(c), cfg_data.get('tv_export', ''), oos_trades_json,
        ))

    # ──────────────────────────────────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────────────────────────────────

    def get_runs(self, instrument: Optional[str] = None,
                 limit: int = 100) -> List[Dict]:
        """Get list of optimization runs."""
        with self._conn() as conn:
            if instrument:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE instrument = ? ORDER BY created_at DESC LIMIT ?",
                    (instrument, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_configs(self, run_id: Optional[int] = None,
                    instrument: Optional[str] = None,
                    oos_gate_only: bool = False,
                    min_score: float = 0,
                    limit: int = 500) -> List[Dict]:
        """Query configs with optional filters."""
        clauses = []
        params = []

        if run_id is not None:
            clauses.append("c.run_id = ?")
            params.append(run_id)
        if instrument:
            clauses.append("r.instrument = ?")
            params.append(instrument)
        if oos_gate_only:
            clauses.append("c.oos_gate_pass = 1")
        if min_score > 0:
            clauses.append("c.score >= ?")
            params.append(min_score)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT c.*, r.instrument, r.mode, r.created_at AS run_date,
                       r.n_trials, r.is_start, r.is_end, r.oos_start, r.oos_end
                FROM configs c
                JOIN runs r ON r.id = c.run_id
                {where}
                ORDER BY c.score DESC
                LIMIT ?
            """, params).fetchall()
            return [dict(r) for r in rows]

    def get_config_by_id(self, config_id: int) -> Optional[Dict]:
        """Get a single config with its run info."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT c.*, r.instrument, r.mode, r.created_at AS run_date,
                       r.n_trials, r.is_start, r.is_end, r.oos_start, r.oos_end
                FROM configs c
                JOIN runs r ON r.id = c.run_id
                WHERE c.id = ?
            """, (config_id,)).fetchone()
            return dict(row) if row else None

    def get_best_per_instrument(self, top_n: int = 5) -> Dict[str, List[Dict]]:
        """
        Get the best configs per instrument, ranked by composite QUALITY score
        that balances OOS performance AND robustness equally.

        V5.4: Uses compute_robustness_score() + compute_quality_grade() from metrics.py.
        Includes deduplication by config fingerprint.

        Returns {instrument: [config, config, ...]} with at most top_n per instrument.
        """
        from .metrics import compute_robustness_score, compute_quality_grade
        from .config import TRANSACTION_COSTS, DEFAULT_TRANSACTION_COST

        with self._conn() as conn:
            # Get all instruments
            instruments = [r[0] for r in conn.execute(
                "SELECT DISTINCT r.instrument FROM runs r "
                "JOIN configs c ON c.run_id = r.id ORDER BY r.instrument"
            ).fetchall()]

            from .config import get_oos_dd_limit

            result = {}
            for inst in instruments:
                rows = conn.execute("""
                    SELECT c.*, r.instrument, r.mode, r.created_at AS run_date,
                           r.n_trials, r.is_start, r.is_end, r.oos_start, r.oos_end,
                           r.data_file
                    FROM configs c
                    JOIN runs r ON r.id = c.run_id
                    WHERE r.instrument = ?
                """, (inst,)).fetchall()

                if not rows:
                    continue

                # Adaptive DD threshold for this instrument
                dd_limit = get_oos_dd_limit(inst)

                # Compute quality score for each config
                scored = []
                for r in rows:
                    d = dict(r)
                    perf_quality = 0.0

                    # Recompute OOS gate on-the-fly with adaptive DD threshold
                    oos_calmar_val = d.get('oos_calmar') or 0
                    oos_pf_val = d.get('oos_profit_factor') or 0
                    oos_dd_val = d.get('oos_max_dd') or 0
                    oos_trades_val = d.get('oos_trades') or 0
                    gate_pass = (
                        oos_calmar_val >= 0.25 and
                        oos_pf_val >= 1.2 and
                        oos_dd_val >= dd_limit and
                        oos_trades_val >= 15
                    )
                    d['oos_gate_pass'] = int(gate_pass)
                    d['_dd_limit'] = dd_limit

                    # ── OOS Performance (perf_quality) ──
                    if gate_pass:
                        perf_quality += 5.0
                    perf_quality += min(max(0, oos_calmar_val), 2.0)

                    # WFE contribution to perf (legacy inline, kept for compat)
                    wfe = d.get('wfe')  # Can be None now (V5.4)
                    if wfe is not None:
                        wfe_capped = min(max(wfe, 0), 1.0)
                        perf_quality += wfe_capped * 2.5
                        # WFE graduated penalty
                        if wfe > 1.5:
                            perf_quality -= 1.5
                        elif wfe > 1.2:
                            perf_quality -= 0.8
                        elif wfe > 1.0:
                            perf_quality -= 0.3
                    # else: no WFE contribution (0 points, no penalty)

                    # Walk-Forward stability (0-100 → 0-2.0)
                    wf_stab = d.get('wf_stability') or 0
                    perf_quality += (min(wf_stab, 100) / 100.0) * 2.0

                    # Walk-Forward consistency (0-100 → 0-0.5)
                    wf_cons = d.get('wf_consistency') or 0
                    perf_quality += (min(wf_cons, 100) / 100.0) * 0.5

                    # Perturbation bonus
                    if d.get('perturbation_stable'):
                        perf_quality += 0.5

                    # IS Score (tiebreaker)
                    perf_quality += (d.get('score') or 0) * 0.2

                    # ── V5.4: Robustness Score + Quality Grade ──
                    robustness = compute_robustness_score(
                        wfe=wfe,
                        perturbation_stable=bool(d.get('perturbation_stable')),
                        perturbation_degradation=d.get('perturbation_degradation'),
                        wf_stability=d.get('wf_stability'),
                        wf_consistency=d.get('wf_consistency'),
                        wf_degradation=bool(d.get('wf_degradation')),
                    )

                    grade_info = compute_quality_grade(
                        perf_score=perf_quality,
                        robustness_score=robustness,
                        entry_type=d.get('entry_type', ''),
                        instrument=inst,
                        transaction_costs=TRANSACTION_COSTS,
                        default_cost=DEFAULT_TRANSACTION_COST,
                    )

                    d['_quality'] = grade_info['quality_score']
                    d['_grade'] = grade_info['grade']
                    d['_robustness'] = grade_info['robustness_score']
                    d['_flags'] = grade_info['flags']
                    d['_perf_quality'] = round(perf_quality, 4)
                    scored.append(d)

                # C5: Deduplication by config fingerprint before sort
                seen_fps = set()
                deduped = []
                for d in scored:
                    cfg_json = d.get('config_json')
                    if cfg_json:
                        try:
                            cfg = json.loads(cfg_json) if isinstance(cfg_json, str) else cfg_json
                            fp = _config_fingerprint_for_dedup(cfg)
                            if fp in seen_fps:
                                continue
                            seen_fps.add(fp)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    deduped.append(d)

                # Sort by quality descending
                deduped.sort(key=lambda x: x['_quality'], reverse=True)
                result[inst] = deduped[:top_n]

            return result

    def get_instruments_summary(self) -> List[Dict]:
        """Get summary per instrument: count of runs, configs, best OOS calmar, etc."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    r.instrument,
                    COUNT(DISTINCT r.id) AS n_runs,
                    COUNT(c.id) AS n_configs,
                    SUM(CASE WHEN c.oos_gate_pass = 1 THEN 1 ELSE 0 END) AS n_oos_pass,
                    MAX(c.oos_calmar) AS best_oos_calmar,
                    MAX(c.oos_profit_factor) AS best_oos_pf,
                    MIN(c.oos_max_dd) AS worst_oos_dd,
                    MAX(c.wfe) AS best_wfe,
                    MAX(c.wf_stability) AS best_wf_stability,
                    MAX(r.created_at) AS last_run
                FROM runs r
                JOIN configs c ON c.run_id = r.id
                GROUP BY r.instrument
                ORDER BY n_oos_pass DESC, best_oos_calmar DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def get_explored_configs(self, instrument: str,
                             data_file: str = '',
                             keys: tuple = ('entry_type', 'pmax_length',
                                            'pmax_multiplier', 'pmax_ma_type',
                                            'pmax_ma_mode', 'pmax_ma_factor',
                                            'pmax_ma_distance',
                                            'use_ma_cross_signal',
                                            'sig_ma_type', 'sig_ma_fast',
                                            'sig_ma_slow',
                                            'don_length', 'rsi_div_length',
                                            'rsi_div_pivot', 'rsi_div_hidden',
                                            'fractal_buffer', 'length_bb_ma',
                                            'mult_bb_ma', 'length_bb_sma',
                                            'use_lt_filter', 'lt_multiplier',
                                            'use_mt_filter',
                                            'use_rsi50_filter',
                                            'use_diff_ma_red',
                                            'diff_ma_red_pct')) -> list:
        """
        Extract all previously explored parameter combinations for a given
        instrument from the database.

        Returns a list of dicts, one per stored config, with only the
        parameter keys relevant for deduplication.  Each dict also carries
        a compact string `_fingerprint` (sorted key=value pairs) that can
        be compared cheaply at trial-evaluation time.

        Used by the optimizer to penalise or skip configurations that are
        too similar to what has already been explored.
        """
        with self._conn() as conn:
            # Scope exclusion to instrument + same data file (= same timeframe)
            # data_file stored in runs table contains the filename which encodes
            # the timeframe (e.g., 'UK100_4H.csv').  If data_file is provided,
            # only exclude configs from runs that used the SAME data file.
            if data_file:
                # Match by filename only (strip directory path)
                import os as _os
                data_basename = _os.path.basename(data_file)
                rows = conn.execute("""
                    SELECT c.config_json
                    FROM configs c
                    JOIN runs r ON r.id = c.run_id
                    WHERE r.instrument = ?
                      AND (r.data_file LIKE ? OR r.data_file LIKE ?)
                """, (instrument,
                       f'%{data_basename}',
                       f'%/{data_basename}')).fetchall()
            else:
                rows = conn.execute("""
                    SELECT c.config_json
                    FROM configs c
                    JOIN runs r ON r.id = c.run_id
                    WHERE r.instrument = ?
                """, (instrument,)).fetchall()

        explored = []
        for row in rows:
            raw = row['config_json']
            if not raw:
                continue
            try:
                cfg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            subset = {}
            for k in keys:
                if k in cfg:
                    v = cfg[k]
                    # Normalise small floats to 1-decimal to avoid fp noise
                    if isinstance(v, float):
                        v = round(v, 1)
                    subset[k] = v

            # Deterministic fingerprint for fast comparison
            fp = '|'.join(f'{k}={subset[k]}' for k in sorted(subset))
            subset['_fingerprint'] = fp
            explored.append(subset)

        return explored

    def delete_run(self, run_id: int):
        """Delete a run and all its configs."""
        with self._conn() as conn:
            conn.execute("DELETE FROM configs WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()

    def delete_config(self, config_id: int) -> bool:
        """Delete a single config by ID. Returns True if deleted."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM configs WHERE id = ?", (config_id,))
            conn.commit()
            return cur.rowcount > 0

    # ──────────────────────────────────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────────────────────────────────

    def export_csv(self, instrument: Optional[str] = None,
                   oos_gate_only: bool = False) -> str:
        """Export configs as CSV string."""
        rows = self.get_configs(instrument=instrument, oos_gate_only=oos_gate_only, limit=10000)
        if not rows:
            return ""

        # Column order for CSV (human-readable)
        columns = [
            'run_date', 'instrument', 'mode', 'rank', 'score',
            'entry_type', 'pmax_ma_type', 'pmax_length', 'pmax_multiplier',
            'use_lt_filter', 'lt_tf', 'lt_length', 'lt_ma_type', 'lt_multiplier',
            'use_mt_filter', 'mt_tf', 'mt_length', 'mt_ma_type', 'mt_multiplier',
            'is_trades', 'is_win_rate', 'is_net_return', 'is_max_dd',
            'is_calmar', 'is_sharpe', 'is_sortino', 'is_profit_factor',
            'oos_trades', 'oos_win_rate', 'oos_net_return', 'oos_max_dd',
            'oos_calmar', 'oos_sharpe', 'oos_sortino', 'oos_profit_factor',
            'oos_gate_pass', 'wfe',
            'perturbation_stable', 'perturbation_degradation',
            'wf_stability', 'wf_consistency', 'wf_avg_calmar', 'wf_avg_pf', 'wf_degradation',
            'don_length', 'rsi_div_length', 'fractal_buffer',
            'length_bb_ma', 'mult_bb_ma', 'length_bb_sma',
            'use_rsi50_filter', 'use_diff_ma_red', 'diff_ma_red_pct',
            'sl_mode', 'sl_pct', 'order_pause', 'long_side', 'short_side',
        ]

        lines = [','.join(columns)]
        for row in rows:
            vals = []
            for col in columns:
                v = row.get(col, '')
                if v is None:
                    v = ''
                elif isinstance(v, float):
                    v = f'{v:.6g}'
                else:
                    v = str(v)
                # Escape commas in values
                if ',' in v or '"' in v:
                    v = '"' + v.replace('"', '""') + '"'
                vals.append(v)
            lines.append(','.join(vals))

        return '\n'.join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # FULL JSON EXPORT / IMPORT  (portable backup)
    # ──────────────────────────────────────────────────────────────────────

    def export_full_json(self) -> dict:
        """
        Export the ENTIRE database (all runs + all configs) as a single
        JSON-serialisable dict.  This is the portable backup format:
        copy the resulting file to a new machine, then call import_full_json()
        to restore everything.

        Format:
          {
            "format": "pf_ai_lab_db_v1",
            "exported_at": "2026-02-13T...",
            "runs": [
              {
                <all run columns>,
                "configs": [
                  {<all config columns>},
                  ...
                ]
              },
              ...
            ]
          }
        """
        with self._conn() as conn:
            runs = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
            result_runs = []
            for run in runs:
                run_d = dict(run)
                run_id = run_d['id']
                configs = conn.execute(
                    "SELECT * FROM configs WHERE run_id = ? ORDER BY rank",
                    (run_id,)).fetchall()
                run_d['configs'] = [dict(c) for c in configs]
                result_runs.append(run_d)

        return {
            'format': 'pf_ai_lab_db_v1',
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'runs': result_runs,
        }

    def import_full_json(self, data: dict) -> dict:
        """
        Import runs + configs from a JSON dict produced by export_full_json().
        Uses the original `created_at` timestamp + instrument + n_configs as
        a deduplication key: runs that already exist are SKIPPED (not duplicated).

        Returns:
            {
                'imported_runs': int,
                'imported_configs': int,
                'skipped_runs': int,   (already present)
                'errors': [str, ...]
            }
        """
        if data.get('format') != 'pf_ai_lab_db_v1':
            return {
                'imported_runs': 0, 'imported_configs': 0,
                'skipped_runs': 0,
                'errors': [f"Unknown format: {data.get('format')}. "
                           f"Expected 'pf_ai_lab_db_v1'."],
            }

        imported_runs = 0
        imported_configs = 0
        skipped_runs = 0
        errors = []

        run_cols = [
            'created_at', 'instrument', 'mode', 'n_trials', 'is_ratio',
            'is_start', 'is_end', 'oos_start', 'oos_end',
            'duration_sec', 'n_configs', 'data_file', 'notes',
        ]
        config_cols = [
            'rank', 'score',
            'entry_type', 'pmax_ma_type', 'pmax_length', 'pmax_multiplier',
            'use_lt_filter', 'lt_tf', 'lt_length', 'lt_ma_type', 'lt_multiplier',
            'use_mt_filter', 'mt_tf', 'mt_length', 'mt_ma_type', 'mt_multiplier',
            'use_rsi50_filter', 'use_diff_ma_red', 'diff_ma_red_pct',
            'use_ma_dir_filter', 'use_mavg_filter',
            'don_length', 'rsi_div_length', 'rsi_div_pivot', 'rsi_div_hidden',
            'fractal_buffer', 'length_bb_ma', 'mult_bb_ma', 'length_bb_sma',
            'sl_mode', 'sl_pct', 'order_pause', 'long_side', 'short_side',
            'is_trades', 'is_win_rate', 'is_net_return', 'is_max_dd',
            'is_calmar', 'is_sharpe', 'is_sortino', 'is_profit_factor',
            'is_avg_win', 'is_avg_loss',
            'oos_trades', 'oos_win_rate', 'oos_net_return', 'oos_max_dd',
            'oos_calmar', 'oos_sharpe', 'oos_sortino', 'oos_profit_factor',
            'oos_gate_pass', 'wfe',
            'perturbation_stable', 'perturbation_degradation',
            'wf_stability', 'wf_consistency', 'wf_avg_calmar', 'wf_avg_pf',
            'wf_degradation',
            'config_json', 'tv_export', 'oos_trades_json',
        ]

        with self._conn() as conn:
            # Build set of existing runs for dedup
            existing = set()
            for r in conn.execute("SELECT created_at, instrument, n_configs FROM runs").fetchall():
                existing.add((r['created_at'], r['instrument'], r['n_configs']))

            for run_data in data.get('runs', []):
                try:
                    dedup_key = (
                        run_data.get('created_at', ''),
                        run_data.get('instrument', ''),
                        run_data.get('n_configs', 0),
                    )
                    if dedup_key in existing:
                        skipped_runs += 1
                        continue

                    # Insert run
                    vals = [run_data.get(c) for c in run_cols]
                    placeholders = ', '.join(['?'] * len(run_cols))
                    col_names = ', '.join(run_cols)
                    cur = conn.execute(
                        f"INSERT INTO runs ({col_names}) VALUES ({placeholders})",
                        vals)
                    new_run_id = cur.lastrowid
                    imported_runs += 1

                    # Insert configs
                    for cfg in run_data.get('configs', []):
                        cfg_vals = [cfg.get(c) for c in config_cols]
                        cfg_placeholders = ', '.join(['?'] * (len(config_cols) + 1))
                        cfg_col_names = 'run_id, ' + ', '.join(config_cols)
                        conn.execute(
                            f"INSERT INTO configs ({cfg_col_names}) "
                            f"VALUES ({cfg_placeholders})",
                            [new_run_id] + cfg_vals)
                        imported_configs += 1

                except Exception as e:
                    errors.append(f"Run {run_data.get('instrument', '?')} "
                                  f"({run_data.get('created_at', '?')}): {e}")

            conn.commit()

        return {
            'imported_runs': imported_runs,
            'imported_configs': imported_configs,
            'skipped_runs': skipped_runs,
            'errors': errors,
        }

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        with self._conn() as conn:
            total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            total_configs = conn.execute("SELECT COUNT(*) FROM configs").fetchone()[0]
            instruments = [r[0] for r in conn.execute(
                "SELECT DISTINCT instrument FROM runs ORDER BY instrument").fetchall()]
            oos_pass = conn.execute(
                "SELECT COUNT(*) FROM configs WHERE oos_gate_pass = 1").fetchone()[0]
            return {
                'total_runs': total_runs,
                'total_configs': total_configs,
                'instruments': instruments,
                'oos_pass_count': oos_pass,
            }
