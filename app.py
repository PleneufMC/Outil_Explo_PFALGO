#!/usr/bin/env python3
"""
PF AI Lab 5.4.8 — Web Dashboard (Flask)
4-page interface for Exploration, Validation, and Portfolio analysis.
"""

import os
import json
import time
import traceback
import threading
from io import StringIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, send_from_directory)


def _sanitize_for_json(obj):
    """
    Recursively convert numpy types to Python-native types for JSON serialization.
    Handles np.float64, np.int64, np.bool_, np.ndarray, pd.Timestamp,
    and replaces NaN/Inf with None to prevent JSON serialization errors.
    """
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    elif isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

from pf_ma_optimizer.data_loader import load_data, load_tv_trades, load_tv_trades_smart
from pf_ma_optimizer.optimizer import run_optimization
from pf_ma_optimizer.tv_export import format_all_configs, format_config_for_tv
from pf_ma_optimizer.tv_settings_parser import (parse_pine_json,
                                                  parse_alert_with_config,
                                                  generate_tv_mapping_guide,
                                                  generate_pine_export_block,
                                                  generate_pine_alert_block,
                                                  generate_pine_patch_instructions)
from pf_ma_optimizer.metrics import compute_metrics, passes_oos_gates, compute_composite_score
from pf_ma_optimizer.backtest_engine import run_backtest
from pf_ma_optimizer.ml.monte_carlo import run_monte_carlo, classify_config
from pf_ma_optimizer.ml.walk_forward import run_walk_forward, format_walk_forward_report
from pf_ma_optimizer.config import (TRANSACTION_COSTS, INSTRUMENT_GROUPS,
                                     register_custom_instrument, LT_TIMEFRAMES, MT_TIMEFRAMES)
from pf_ma_optimizer.backtest_db import BacktestDB

app = Flask(__name__)
app.secret_key = 'pfalgo_v691_secret_key_2026'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Build info — auto-generated from app startup timestamp
APP_VERSION = '5.4.8'
_BUILD_TS = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')
BUILD_NUMBER = f'{APP_VERSION}-b{_BUILD_TS}'


@app.context_processor
def inject_build_info():
    """Inject version and build number into all templates."""
    return {
        'app_version': APP_VERSION,
        'build_number': BUILD_NUMBER,
        'lt_timeframes': LT_TIMEFRAMES,
        'mt_timeframes': MT_TIMEFRAMES,
    }

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Backtest results database (SQLite)
backtest_db = BacktestDB()


def _resolve_instrument(form):
    """
    Resolve instrument name from form data.
    If instrument == '__custom__', use custom_instrument_name and register it
    with the optional custom transaction cost.
    Returns the instrument name string.
    """
    instrument = form.get('instrument', 'UK100')
    if instrument == '__custom__':
        custom_name = (form.get('custom_instrument_name', '') or '').strip().upper()
        if not custom_name:
            custom_name = 'CUSTOM'
        cost_str = form.get('custom_transaction_cost', '')
        cost = None
        if cost_str:
            try:
                cost = float(cost_str)
            except (ValueError, TypeError):
                pass
        register_custom_instrument(custom_name, cost)
        instrument = custom_name
    return instrument

# Global state for async optimization
optimization_state = {
    'running': False,
    'progress': 0,
    'total': 0,
    'best_score': 0,
    'result': None,
    'error': None,
}


# ==============================================================================
# Routes
# ==============================================================================

@app.route('/api/health')
def health_check():
    """Health-check endpoint used by the launcher to confirm Flask is ready.
    Returns 200 with a JSON body — this is the ONLY reliable way to know
    that Flask has fully initialised (templates loaded, DB connected, etc.).
    """
    return jsonify({
        'status': 'ok',
        'version': APP_VERSION,
        'build': BUILD_NUMBER,
    })


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No favicon, prevent 404

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['GET', 'POST'])
def data_page():
    """Page 1: Data Upload & Preview"""
    if request.method == 'POST':
        file = request.files.get('datafile')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                df, source_type = load_data(filepath)

                # Build preview data — convert Timestamps to strings for JSON
                head_df = df.head(10).reset_index()
                for c in head_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(head_df[c]):
                        head_df[c] = head_df[c].astype(str)

                ohlc_df = df.tail(100).reset_index()
                for c in ohlc_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(ohlc_df[c]):
                        ohlc_df[c] = ohlc_df[c].astype(str)

                preview = {
                    'filename': file.filename,
                    'filepath': filepath,
                    'source_type': source_type,
                    'n_bars': len(df),
                    'date_start': str(df.index[0]),
                    'date_end': str(df.index[-1]),
                    'columns': list(df.columns),
                    'head': head_df.to_dict('records'),
                    'ohlc_sample': ohlc_df.to_dict('records'),
                }
                session['data_filepath'] = filepath
                session['data_filename'] = file.filename
                return render_template('data.html', preview=preview,
                                     instrument_groups=INSTRUMENT_GROUPS)
            except Exception as e:
                import traceback
                err_detail = f"{str(e)}\n\n{traceback.format_exc()}"
                return render_template('data.html', error=err_detail,
                                     instrument_groups=INSTRUMENT_GROUPS)

    return render_template('data.html', instrument_groups=INSTRUMENT_GROUPS)


@app.route('/explore', methods=['GET', 'POST'])
def explore_page():
    """Page 2: Run Exploration"""
    if request.method == 'POST':
        filepath = request.form.get('filepath') or session.get('data_filepath')
        instrument = _resolve_instrument(request.form)
        n_trials = int(request.form.get('n_trials', 500))
        mode = request.form.get('mode', 'focused')
        is_ratio = float(request.form.get('is_ratio', 0.70))
        exclude_db = request.form.get('exclude_db', '0') == '1'

        # Entry signal selection
        entry_map = {
            'entry_Donc_Chikou': 'Donc+Chikou',
            'entry_RSI_Divergence': 'RSI Divergence',
            'entry_Fractal': 'Fractal',
            'entry_Boll_SMA': 'Boll+SMA',
            'entry_MM_Cross': 'MM Cross',
        }
        selected_entries = [v for k, v in entry_map.items()
                           if request.form.get(k, '0') == '1']
        if not selected_entries:
            selected_entries = list(entry_map.values())  # fallback: all
        entry_quota = request.form.get('entry_quota', '0') == '1'

        # Lock parameters — user can fix specific values
        locked_params = {}
        lock_defs = {
            'lock_pmax_length':         ('pmax_length',         'int'),
            'lock_pmax_multiplier':     ('pmax_multiplier',     'float'),
            'lock_pmax_ma_mode':        ('pmax_ma_mode',        'str'),
            'lock_entry_type':          ('entry_type',          'str'),
            'lock_use_lt_filter':       ('use_lt_filter',       'bool'),
            'lock_lt_multiplier':       ('lt_multiplier',       'float'),
            'lock_use_mt_filter':       ('use_mt_filter',       'bool'),
            'lock_use_rsi50_filter':    ('use_rsi50_filter',    'bool'),
            'lock_use_diff_ma_red':     ('use_diff_ma_red',     'bool'),
            'lock_don_length':          ('don_length',          'int'),
            'lock_order_pause':         ('order_pause',         'int'),
        }
        for form_key, (param_name, param_type) in lock_defs.items():
            if request.form.get(form_key, '0') == '1':
                raw_val = request.form.get(form_key + '_val', '')
                if raw_val:
                    if param_type == 'int':
                        locked_params[param_name] = int(float(raw_val))
                    elif param_type == 'float':
                        locked_params[param_name] = float(raw_val)
                    elif param_type == 'bool':
                        locked_params[param_name] = raw_val.lower() in ('true', '1', 'on')
                    else:
                        locked_params[param_name] = raw_val

        if not filepath or not os.path.exists(filepath):
            return render_template('explore.html', error='No data file. Please upload on Data page first.')

        session['instrument'] = instrument
        session['data_filepath'] = filepath

        # Start async optimization
        optimization_state['running'] = True
        optimization_state['progress'] = 0
        optimization_state['total'] = n_trials
        optimization_state['result'] = None
        optimization_state['error'] = None

        def run_async():
            try:
                df, _ = load_data(filepath)

                # Phase indicator for frontend
                # total = n_trials + 20 (OOS) + 20 (perturbation) + 20 (WF) = 60 extra
                # All 20 configs now get perturbation + WF (ProcessPool makes it fast)
                total_with_post = n_trials + 60

                def progress_cb(current, total, best):
                    optimization_state['progress'] = current
                    optimization_state['total'] = total_with_post
                    optimization_state['best_score'] = best

                result = run_optimization(
                    df=df, instrument=instrument,
                    n_trials=n_trials, mode=mode,
                    is_ratio=is_ratio, progress_callback=progress_cb,
                    exclude_db_configs=exclude_db,
                    selected_entries=selected_entries,
                    entry_quota=entry_quota,
                    locked_params=locked_params if locked_params else None,
                    data_file=filepath,
                )

                # Signal completion
                optimization_state['progress'] = total_with_post
                optimization_state['total'] = total_with_post

                # Serialize results — sanitize numpy types for JSON
                serialized = _sanitize_for_json({
                    'top_configs': result['top_configs'],
                    'is_range': result['is_range'],
                    'oos_range': result['oos_range'],
                    'duration_sec': result['duration_sec'],
                    'instrument': instrument,
                    'n_excluded_trials': result.get('n_excluded_trials', 0),
                })
                optimization_state['result'] = serialized

                # Auto-save to backtest database
                try:
                    run_id = backtest_db.save_run(
                        serialized, mode=mode, n_trials=n_trials,
                        is_ratio=is_ratio, data_file=filepath or '',
                        notes=f'Auto-saved from web UI ({BUILD_NUMBER})')
                    print(f'[DB] Run saved: id={run_id}, {instrument}, {len(serialized.get("top_configs", []))} configs')
                except Exception as db_err:
                    print(f'[DB WARNING] Failed to save run: {db_err}')
            except Exception as e:
                optimization_state['error'] = str(e) + '\n\n' + traceback.format_exc()
                import sys
                print(f'[run_async ERROR] {e}', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            finally:
                optimization_state['running'] = False

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        return render_template('explore.html', running=True, n_trials=n_trials,
                             instrument=instrument, mode=mode,
                             exclude_db=exclude_db,
                             instrument_groups=INSTRUMENT_GROUPS)

    # Smart default: pre-check "Exclude DB" if DB already has configs for
    # the last-used instrument (or default UK100).  This way the user never
    # accidentally re-explores the same parameter space.
    instrument_hint = session.get('instrument', 'UK100')
    data_file_hint = session.get('data_filepath', '')
    db_config_count = 0
    try:
        explored = backtest_db.get_explored_configs(instrument_hint, data_file=data_file_hint)
        db_config_count = len({c['_fingerprint'] for c in explored})
    except Exception:
        pass

    # DB stats for the summary line
    try:
        db_stats = backtest_db.get_stats()
    except Exception:
        db_stats = {'total_runs': 0, 'total_configs': 0, 'instruments': [], 'oos_pass_count': 0}

    return render_template('explore.html',
                         filepath=session.get('data_filepath', ''),
                         instrument_groups=INSTRUMENT_GROUPS,
                         smart_exclude_db=db_config_count > 0,
                         smart_exclude_count=db_config_count,
                         smart_instrument=instrument_hint,
                         db_stats=db_stats)


@app.route('/api/db/instrument_config_count')
def api_db_instrument_config_count():
    """API: return the number of unique configs stored in DB for a given instrument+data_file."""
    instrument = request.args.get('instrument', '')
    data_file = request.args.get('data_file', '') or session.get('data_filepath', '')
    if not instrument:
        return jsonify({'count': 0, 'instrument': ''})
    try:
        explored = backtest_db.get_explored_configs(instrument, data_file=data_file)
        unique_fps = {c['_fingerprint'] for c in explored}
        return jsonify({'count': len(unique_fps), 'instrument': instrument})
    except Exception as e:
        return jsonify({'count': 0, 'instrument': instrument, 'error': str(e)})


@app.route('/api/optimization_reset', methods=['POST'])
def optimization_reset():
    """API: clear optimization state so the form is shown again."""
    optimization_state['running'] = False
    optimization_state['progress'] = 0
    optimization_state['total'] = 0
    optimization_state['best_score'] = 0
    optimization_state['result'] = None
    optimization_state['error'] = None
    return jsonify({'ok': True})


@app.route('/api/optimization_status')
def optimization_status():
    """API endpoint for polling optimization progress."""
    return jsonify({
        'running': optimization_state['running'],
        'progress': optimization_state['progress'],
        'total': optimization_state['total'],
        'best_score': optimization_state['best_score'],
        'has_result': optimization_state['result'] is not None,
        'error': optimization_state['error'],
    })


@app.route('/api/optimization_result')
def optimization_result():
    """API endpoint to get optimization results."""
    result = optimization_state['result']
    if not result:
        return jsonify({'error': 'No results available'})

    # Clean up for JSON serialization
    configs = []
    for cfg in result.get('top_configs', []):
        clean_metrics_is = {k: v for k, v in cfg.get('metrics_is', {}).items()
                          if k not in ('equity_curve', 'equity_abs', 'drawdown_curve')}
        clean_metrics_oos = {k: v for k, v in cfg.get('metrics_oos', {}).items()
                           if k not in ('equity_curve', 'equity_abs', 'drawdown_curve')}

        # Clean WF windows metrics (remove equity curves)
        wf_data = cfg.get('walk_forward')
        if wf_data and 'windows' in wf_data:
            clean_wf = {k: v for k, v in wf_data.items() if k != 'windows'}
            clean_wf['n_windows'] = len(wf_data.get('windows', []))
        else:
            clean_wf = wf_data

        configs.append({
            'rank': cfg['rank'],
            'score': cfg['score'],
            'config': cfg['config'],
            'metrics_is': clean_metrics_is,
            'metrics_oos': clean_metrics_oos,
            'tv_export': format_config_for_tv(cfg),
            'perturbation_pass': cfg.get('perturbation_pass'),
            'walk_forward': clean_wf,
        })

    return jsonify({
        'configs': configs,
        'is_range': result.get('is_range'),
        'oos_range': result.get('oos_range'),
        'duration_sec': result.get('duration_sec'),
        'instrument': result.get('instrument'),
        'n_excluded_trials': result.get('n_excluded_trials', 0),
    })


# ==============================================================================
# BACKTEST DATABASE — Page + API
# ==============================================================================

@app.route('/database')
def database_page():
    """Page: Backtest Database raw browser."""
    stats = backtest_db.get_stats()
    return render_template('database.html',
                         stats=stats,
                         instrument_groups=INSTRUMENT_GROUPS)


@app.route('/best')
def best_configs_page():
    """Page 5: Best theoretical configs per instrument."""
    stats = backtest_db.get_stats()
    instruments_summary = backtest_db.get_instruments_summary()
    return render_template('best.html',
                         stats=stats,
                         instruments_summary=instruments_summary,
                         instrument_groups=INSTRUMENT_GROUPS)


@app.route('/api/db/best')
def api_db_best():
    """API: get best configs per instrument.
    V5.4: includes _grade, _quality, _robustness, _flags from Quality Score V2.
    """
    top_n = request.args.get('top_n', 5, type=int)
    instrument = request.args.get('instrument', '')
    best = backtest_db.get_best_per_instrument(top_n=top_n)
    if instrument and instrument in best:
        best = {instrument: best[instrument]}
    # Strip heavy fields for lighter response (keep data_file for TF detection)
    for inst, configs in best.items():
        for c in configs:
            c.pop('config_json', None)
            c.pop('tv_export', None)
            c.pop('oos_trades_json', None)
    return jsonify({'best': best, 'instruments': list(best.keys())})


@app.route('/api/db/runs')
def api_db_runs():
    """API: list optimization runs."""
    instrument = request.args.get('instrument', '')
    runs = backtest_db.get_runs(instrument=instrument or None, limit=200)
    return jsonify({'runs': runs})


@app.route('/api/db/configs')
def api_db_configs():
    """API: query configs with filters."""
    run_id = request.args.get('run_id', type=int)
    instrument = request.args.get('instrument', '')
    oos_only = request.args.get('oos_gate_only', '0') == '1'
    min_score = request.args.get('min_score', 0, type=float)
    configs = backtest_db.get_configs(
        run_id=run_id, instrument=instrument or None,
        oos_gate_only=oos_only, min_score=min_score, limit=500)
    # Strip config_json and tv_export for lighter response
    for c in configs:
        c.pop('config_json', None)
        c.pop('tv_export', None)
    return jsonify({'configs': configs, 'total': len(configs)})


@app.route('/api/db/config/<int:config_id>')
def api_db_config_detail(config_id):
    """API: get single config with full JSON + TV export."""
    cfg = backtest_db.get_config_by_id(config_id)
    if not cfg:
        return jsonify({'error': 'Config not found'}), 404
    # Strip oos_trades_json from detail (use dedicated endpoint)
    cfg.pop('oos_trades_json', None)
    return jsonify(cfg)


@app.route('/api/db/config/<int:config_id>/trades')
def api_db_config_trades(config_id):
    """API: get OOS trades for a specific config (C4)."""
    cfg = backtest_db.get_config_by_id(config_id)
    if not cfg:
        return jsonify({'error': 'Config not found'}), 404
    trades_json = cfg.get('oos_trades_json')
    if not trades_json:
        return jsonify({'trades': [], 'count': 0, 'config_id': config_id})
    try:
        trades = json.loads(trades_json) if isinstance(trades_json, str) else trades_json
    except (json.JSONDecodeError, TypeError):
        trades = []
    return jsonify({'trades': trades, 'count': len(trades), 'config_id': config_id})


@app.route('/api/db/export_csv')
def api_db_export_csv():
    """API: export all configs as CSV download."""
    instrument = request.args.get('instrument', '')
    oos_only = request.args.get('oos_gate_only', '0') == '1'
    csv_content = backtest_db.export_csv(
        instrument=instrument or None, oos_gate_only=oos_only)
    if not csv_content:
        return jsonify({'error': 'No data to export'}), 404
    from flask import Response
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=backtest_db_{instrument or "all"}_{datetime.now().strftime("%Y%m%d")}.csv'}
    )


@app.route('/api/db/export_json')
def api_db_export_json():
    """API: export ENTIRE database as a portable JSON file (runs + configs)."""
    data = backtest_db.export_full_json()
    json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    from flask import Response
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    return Response(
        json_str,
        mimetype='application/json',
        headers={
            'Content-Disposition': f'attachment; filename=pf_ai_lab_db_{ts}.json'
        }
    )


@app.route('/api/db/import_json', methods=['POST'])
def api_db_import_json():
    """API: import runs + configs from a JSON backup file."""
    file = request.files.get('db_file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        raw = file.read().decode('utf-8')
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return jsonify({'error': f'Invalid JSON file: {e}'}), 400

    result = backtest_db.import_full_json(data)
    return jsonify(result)


@app.route('/api/db/delete_run/<int:run_id>', methods=['DELETE'])
def api_db_delete_run(run_id):
    """API: delete a run and all its configs."""
    backtest_db.delete_run(run_id)
    return jsonify({'ok': True, 'deleted_run_id': run_id})


@app.route('/api/db/delete_config/<int:config_id>', methods=['DELETE'])
def api_db_delete_config(config_id):
    """API: delete a single config by ID."""
    deleted = backtest_db.delete_config(config_id)
    if deleted:
        return jsonify({'ok': True, 'deleted_config_id': config_id})
    return jsonify({'error': 'Config not found'}), 404


@app.route('/api/db/save_current', methods=['POST'])
def api_db_save_current():
    """API: manually save current optimization results to DB."""
    result = optimization_state.get('result')
    if not result:
        return jsonify({'error': 'No optimization results to save'}), 400

    mode = request.json.get('mode', 'focused') if request.is_json else 'focused'
    notes = request.json.get('notes', '') if request.is_json else ''
    n_trials = request.json.get('n_trials', 500) if request.is_json else 500

    # Add tv_export to each config if missing
    for cfg_data in result.get('top_configs', []):
        if 'tv_export' not in cfg_data:
            cfg_data['tv_export'] = format_config_for_tv(cfg_data)

    run_id = backtest_db.save_run(result, mode=mode, n_trials=n_trials, notes=notes)
    return jsonify({'ok': True, 'run_id': run_id})


@app.route('/validate', methods=['GET', 'POST'])
def validate_page():
    """Page 3: Post-TV Validation (Monte Carlo + Walk-Forward)"""
    if request.method == 'POST':
        file = request.files.get('tv_export')
        mc_sims = int(request.form.get('mc_sims', 10000))
        ruin_threshold = float(request.form.get('ruin_threshold', -20))

        # Optional: OHLCV data + config for Walk-Forward
        ohlcv_file = request.files.get('ohlcv_data')
        instrument = request.form.get('instrument', 'UK100')
        config_json = request.form.get('config_json', '').strip()

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, 'tv_' + file.filename)
            file.save(filepath)

            try:
                # Smart parsing: auto-detects correct sheet + filters Exit rows
                pnls_pct, pnls_usd, n_trades, parse_info = load_tv_trades_smart(filepath)

                # Prefer P&L percentages for Monte Carlo (scale-independent)
                if len(pnls_pct) > 0:
                    pnls = pnls_pct
                    pnl_type = 'pct'
                elif len(pnls_usd) > 0:
                    pnls = pnls_usd
                    pnl_type = 'usd'
                else:
                    return render_template('validate.html',
                                         error=(
                                             f'Could not identify P&L column in the file.\n'
                                             f'Sheet used: {parse_info["sheet_used"]}\n'
                                             f'Columns found: {parse_info["columns"]}\n'
                                             f'Please ensure the file contains a "List of trades" sheet '
                                             f'with Net P&L columns.'
                                         ),
                                         instrument_groups=INSTRUMENT_GROUPS)

                if len(pnls) < 10:
                    return render_template('validate.html',
                                         error=(
                                             f'Only {len(pnls)} trades found (minimum 10 required).\n'
                                             f'Sheet: {parse_info["sheet_used"]}, '
                                             f'Rows: {parse_info["total_rows"]}, '
                                             f'Exit rows: {parse_info["exit_rows"]}'
                                         ),
                                         instrument_groups=INSTRUMENT_GROUPS)

                # Run Monte Carlo
                mc_result = run_monte_carlo(pnls, n_sims=mc_sims,
                                           ruin_threshold=ruin_threshold)
                mc_verdict = classify_config(mc_result)

                # Build histogram data (adaptive bins to avoid ValueError
                # when data range is too narrow for 50 bins)
                def _safe_histogram(data, target_bins=50):
                    arr = np.asarray(data, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if len(arr) < 2:
                        return (np.array([len(arr)]), np.array([0.0, 1.0]))
                    n_unique = len(np.unique(arr))
                    bins = min(target_bins, max(1, n_unique))
                    return np.histogram(arr, bins=bins)

                dd_hist = _safe_histogram(mc_result['max_dd_dist'], 50)
                ret_hist = _safe_histogram(mc_result['return_dist'], 50)

                # Trade stats
                total_pnl_display = round(float(np.sum(pnls_usd)), 2) if len(pnls_usd) > 0 else round(float(np.sum(pnls)), 2)
                win_rate = round(float(np.mean(pnls > 0) * 100), 1)

                # --- Walk-Forward (optional, requires OHLCV + config) ---
                wf_result = None
                wf_error = None

                if ohlcv_file and ohlcv_file.filename and config_json:
                    try:
                        ohlcv_path = os.path.join(UPLOAD_FOLDER, 'wf_' + ohlcv_file.filename)
                        ohlcv_file.save(ohlcv_path)
                        df_ohlcv, _ = load_data(ohlcv_path)

                        config = json.loads(config_json)

                        wf_raw = run_walk_forward(
                            df=df_ohlcv, config=config, instrument=instrument,
                            n_windows=5, is_pct=0.60, overlap=0.5
                        )

                        # Clean window metrics for display
                        wf_windows = []
                        for w in wf_raw.get('windows', []):
                            m_oos = w.get('metrics_oos', {})
                            wf_windows.append({
                                'window': w['window'],
                                'start': w['start'][:10],
                                'end': w['end'][:10],
                                'n_bars': w['n_bars'],
                                'oos_trades': m_oos.get('n_trades', 0),
                                'oos_calmar': round(m_oos.get('calmar', 0), 3),
                                'oos_pf': round(m_oos.get('profit_factor', 0), 3),
                                'oos_dd': round(m_oos.get('max_dd', 0), 2),
                                'oos_wr': round(m_oos.get('win_rate', 0), 1),
                            })

                        wf_result = {
                            'stability_score': wf_raw['stability_score'],
                            'consistency': wf_raw['consistency'],
                            'avg_calmar': wf_raw['avg_calmar'],
                            'avg_pf': wf_raw['avg_pf'],
                            'degradation': wf_raw['degradation'],
                            'n_windows': len(wf_windows),
                            'windows': wf_windows,
                        }
                    except json.JSONDecodeError:
                        wf_error = 'Invalid JSON in config field.'
                    except Exception as e:
                        wf_error = f'Walk-Forward error: {str(e)}'

                # --- Combined verdict (MC + WF) ---
                combined_verdict = _compute_combined_verdict(mc_verdict, wf_result)

                validation = {
                    'filename': file.filename,
                    'n_trades': n_trades,
                    'pnl_type': pnl_type,
                    'mc': mc_result,
                    'mc_verdict': mc_verdict,
                    'verdict': combined_verdict,
                    'dd_histogram': {
                        'bins': dd_hist[1].tolist(),
                        'counts': dd_hist[0].tolist(),
                    },
                    'ret_histogram': {
                        'bins': ret_hist[1].tolist(),
                        'counts': ret_hist[0].tolist(),
                    },
                    'trade_stats': {
                        'total_pnl': total_pnl_display,
                        'total_pnl_pct': round(float(np.sum(pnls_pct)), 2) if len(pnls_pct) > 0 else None,
                        'win_rate': win_rate,
                        'avg_win': round(float(np.mean(pnls[pnls > 0])), 4) if np.any(pnls > 0) else 0,
                        'avg_loss': round(float(np.mean(pnls[pnls < 0])), 4) if np.any(pnls < 0) else 0,
                    },
                    'parse_info': {
                        'sheet': parse_info['sheet_used'],
                        'exit_rows': parse_info['exit_rows'],
                        'total_rows': parse_info['total_rows'],
                    },
                    'walk_forward': wf_result,
                    'wf_error': wf_error,
                }

                return render_template('validate.html', validation=validation,
                                     instrument_groups=INSTRUMENT_GROUPS)

            except Exception as e:
                import traceback
                return render_template('validate.html',
                                     error=f"{str(e)}\n\n{traceback.format_exc()}",
                                     instrument_groups=INSTRUMENT_GROUPS)

    return render_template('validate.html',
                         instrument_groups=INSTRUMENT_GROUPS)


def _compute_combined_verdict(mc_verdict: dict, wf_result: dict = None) -> dict:
    """
    Combine Monte Carlo and Walk-Forward results into a final verdict.

    Rules:
    - MC alone → use MC verdict (as before)
    - MC + WF → WF can DOWNGRADE the verdict but never upgrade it
      - MC=DEPLOY + WF>=70 → DEPLOY
      - MC=DEPLOY + WF 40-69 → MONITOR (downgraded)
      - MC=DEPLOY + WF<40 → MONITOR (downgraded)
      - MC=MONITOR + WF>=40 → MONITOR
      - MC=MONITOR + WF<40 → REJECT (downgraded)
      - MC=REJECT → always REJECT
    """
    verdict = mc_verdict['verdict']
    confidence = mc_verdict['confidence']
    reasons = list(mc_verdict.get('reasons', []))

    if wf_result is None:
        reasons.append('Walk-Forward: not run (no OHLCV data + config provided)')
        return {
            'verdict': verdict,
            'classification': mc_verdict.get('classification', 'UNKNOWN'),
            'confidence': confidence,
            'reasons': reasons,
            'p_ruin': mc_verdict.get('p_ruin', 100),
            'has_wf': False,
        }

    wf_score = wf_result.get('stability_score', 0)
    wf_consistency = wf_result.get('consistency', 0)
    wf_degradation = wf_result.get('degradation', False)

    # WF can only downgrade, never upgrade
    if verdict == 'DEPLOY':
        if wf_score >= 70 and not wf_degradation:
            reasons.append(f'Walk-Forward: STABLE ({wf_score:.0f}/100) — confirms deployment')
        elif wf_score >= 40:
            verdict = 'MONITOR'
            confidence = max(0, confidence - 20)
            reasons.append(f'Walk-Forward: MODERATE ({wf_score:.0f}/100) — downgraded to MONITOR')
        else:
            verdict = 'MONITOR'
            confidence = max(0, confidence - 30)
            reasons.append(f'Walk-Forward: UNSTABLE ({wf_score:.0f}/100) — downgraded to MONITOR')
    elif verdict == 'MONITOR':
        if wf_score >= 40 and not wf_degradation:
            reasons.append(f'Walk-Forward: OK ({wf_score:.0f}/100) — maintains MONITOR')
        else:
            verdict = 'REJECT'
            confidence = max(0, confidence - 20)
            reasons.append(f'Walk-Forward: UNSTABLE ({wf_score:.0f}/100) — downgraded to REJECT')
    else:
        reasons.append(f'Walk-Forward: {wf_score:.0f}/100 (MC already REJECT)')

    if wf_degradation:
        reasons.append('Walk-Forward: performance DEGRADING over time')

    if wf_consistency < 50:
        reasons.append(f'Walk-Forward: only {wf_consistency:.0f}% of windows profitable')

    return {
        'verdict': verdict,
        'classification': mc_verdict.get('classification', 'UNKNOWN'),
        'confidence': round(max(0, min(100, confidence)), 1),
        'reasons': reasons,
        'p_ruin': mc_verdict.get('p_ruin', 100),
        'has_wf': True,
        'wf_score': wf_score,
    }


# ==============================================================================
# ROBUSTNESS TEST — Dedicated page for testing a single config
# ==============================================================================

def _build_config_from_form(form):
    """
    Build a complete config dict from the visual form fields.
    Mirrors get_default_config() structure from config.py.
    Handles type conversion (str→int/float/bool) for all fields.
    """
    def _int(key, default=0):
        try: return int(float(form.get(key, default)))
        except (ValueError, TypeError): return default

    def _float(key, default=0.0):
        try: return float(form.get(key, default))
        except (ValueError, TypeError): return default

    def _bool(key, default=False):
        val = form.get(key, '')
        if val.lower() in ('true', '1', 'on'):
            return True
        elif val.lower() in ('false', '0', 'off'):
            return False
        return default

    def _str(key, default=''):
        return form.get(key, default) or default

    from pf_ma_optimizer.config import get_default_config
    config = get_default_config()  # start from complete defaults

    # PMax
    config['pmax_ma_type'] = _str('pmax_ma_type', 'EMA')
    config['pmax_length'] = _int('pmax_length', 10)
    config['pmax_multiplier'] = _float('pmax_multiplier', 3.0)
    config['pmax_ma_mode'] = _str('pmax_ma_mode', 'classic')
    config['pmax_ma_factor'] = _float('pmax_ma_factor', 1.0)
    config['pmax_ma_distance'] = _float('pmax_ma_distance', 0.0)

    # Entry
    config['entry_type'] = _str('entry_type', 'Donc+Chikou')
    config['order_pause'] = _int('order_pause', 5)
    config['don_length'] = _int('don_length', 20)
    config['rsi_div_length'] = _int('rsi_div_length', 14)
    config['rsi_div_pivot'] = _int('rsi_div_pivot', 5)
    config['rsi_div_hidden'] = _bool('rsi_div_hidden', True)
    config['fractal_buffer'] = _float('fractal_buffer', 0.0)
    config['length_bb_ma'] = _int('length_bb_ma', 20)
    config['mult_bb_ma'] = _float('mult_bb_ma', 2.0)
    config['length_bb_sma'] = _int('length_bb_sma', 50)
    config['mm_cross_ma1_type'] = _str('mm_cross_ma1_type', 'EMA')
    config['mm_cross_ma1_len'] = _int('mm_cross_ma1_len', 9)
    config['mm_cross_ma2_type'] = _str('mm_cross_ma2_type', 'EMA')
    config['mm_cross_ma2_len'] = _int('mm_cross_ma2_len', 21)

    # NOTE: use_ma_cross_signal removed from UI — MA crossover is an entry type
    # ("MM Cross"), not a separate filter. The config key still exists in the
    # engine for backward compat but the form no longer exposes it.

    # Filters
    config['use_lt_filter'] = _bool('use_lt_filter', False)
    config['lt_ma_type'] = _str('lt_ma_type', 'EMA')
    config['lt_length'] = _int('lt_length', 20)
    config['lt_tf'] = _str('lt_tf', 'W')
    config['lt_multiplier'] = _float('lt_multiplier', 1.5)

    config['use_mt_filter'] = _bool('use_mt_filter', False)
    config['mt_ma_type'] = _str('mt_ma_type', 'EMA')
    config['mt_length'] = _int('mt_length', 10)
    config['mt_tf'] = _str('mt_tf', '4H')
    config['mt_multiplier'] = _float('mt_multiplier', 1.5)

    config['use_rsi50_filter'] = _bool('use_rsi50_filter', False)
    config['use_diff_ma_red'] = _bool('use_diff_ma_red', False)
    config['diff_ma_red_pct'] = _float('diff_ma_red_pct', 1.0)
    config['use_ma_dir_filter'] = _bool('use_ma_dir_filter', False)
    config['ma_dir_len'] = _int('ma_dir_len', 3)
    config['use_mavg_filter'] = _bool('use_mavg_filter', False)
    config['use_diff_price_red'] = _bool('use_diff_price_red', False)
    config['diff_price_red_pct'] = _float('diff_price_red_pct', 1.0)

    # Risk
    config['sl_mode'] = _str('sl_mode', 'static_pct')
    config['sl_pct'] = _float('sl_pct', 30.0)
    config['atr_sl_mult'] = _float('atr_sl_mult', 2.0)
    config['exit_setup_reversal'] = _bool('exit_setup_reversal', True)
    config['long_side'] = _bool('long_side', True)
    config['short_side'] = _bool('short_side', True)

    # Custom transaction cost (for custom instruments)
    custom_tc = form.get('custom_transaction_cost', '')
    if custom_tc:
        try:
            config['custom_transaction_cost'] = float(custom_tc)
        except (ValueError, TypeError):
            pass

    return config

@app.route('/test', methods=['GET', 'POST'])
def test_page():
    """Page: Robustness Test — visual form (like TradingView) or JSON import."""
    if request.method == 'POST':
        input_mode = request.form.get('input_mode', 'form')
        ohlcv_file = request.files.get('ohlcv_data')
        instrument = _resolve_instrument(request.form)
        is_ratio = float(request.form.get('is_ratio', 0.70))

        # ── Build config from form fields OR JSON ──
        if input_mode == 'json':
            # JSON import mode
            config_json = request.form.get('config_json', '').strip()
            if not config_json:
                return render_template('test.html', error='Le champ JSON est vide.',
                                       instrument_groups=INSTRUMENT_GROUPS,
                                       session_instrument=session.get('instrument', 'UK100'))
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError as e:
                return render_template('test.html', error=f'JSON invalide: {e}',
                                       instrument_groups=INSTRUMENT_GROUPS,
                                       session_instrument=session.get('instrument', 'UK100'),
                                       prefill_config=config_json)
        else:
            # Visual form mode — build config dict from form fields
            config = _build_config_from_form(request.form)

        if not ohlcv_file or not ohlcv_file.filename:
            return render_template('test.html', error='Veuillez uploader un fichier OHLCV.',
                                   instrument_groups=INSTRUMENT_GROUPS,
                                   session_instrument=session.get('instrument', 'UK100'),
                                   prefill_config=json.dumps(config, indent=2))

        try:
            # Save and load OHLCV data
            ohlcv_path = os.path.join(UPLOAD_FOLDER, 'test_' + ohlcv_file.filename)
            ohlcv_file.save(ohlcv_path)
            df, _ = load_data(ohlcv_path)

            # Split IS/OOS
            n = len(df)
            split_idx = int(n * is_ratio)
            if split_idx < 50 or (n - split_idx) < 50:
                return render_template('test.html',
                                       error=f'IS ({split_idx} bars) or OOS ({n - split_idx} bars) trop petit.',
                                       instrument_groups=INSTRUMENT_GROUPS,
                                       session_instrument=instrument,
                                       prefill_config=json.dumps(config, indent=2))

            df_is = df.iloc[:split_idx].copy()
            df_oos = df.iloc[split_idx:].copy()
            is_range = (str(df_is.index[0])[:10], str(df_is.index[-1])[:10])
            oos_range = (str(df_oos.index[0])[:10], str(df_oos.index[-1])[:10])

            # Run IS backtest
            trades_is = run_backtest(df_is, config, instrument)
            metrics_is = compute_metrics(trades_is)

            # Run OOS backtest
            trades_oos = run_backtest(df_oos, config, instrument)
            metrics_oos = compute_metrics(trades_oos)

            # OOS Gate check (adaptive per instrument)
            from pf_ma_optimizer.config import get_oos_dd_limit
            dd_limit = get_oos_dd_limit(instrument)
            oos_gate = {
                'calmar': (metrics_oos.get('calmar', 0) or 0) >= 0.25,
                'pf': (metrics_oos.get('profit_factor', 0) or 0) >= 1.2,
                'dd': (metrics_oos.get('max_dd', 0) or 0) >= dd_limit,
                'dd_limit': dd_limit,
                'trades': (metrics_oos.get('n_trades', 0) or 0) >= 15,
            }
            oos_gate['all_pass'] = all([oos_gate['calmar'], oos_gate['pf'],
                                        oos_gate['dd'], oos_gate['trades']])

            # WFE
            is_calmar = metrics_is.get('calmar', 0) or 0
            oos_calmar = metrics_oos.get('calmar', 0) or 0
            wfe = oos_calmar / is_calmar if is_calmar > 0 else 0

            # Perturbation
            from pf_ma_optimizer.optimizer import _check_perturbation_robustness
            perturbation = _check_perturbation_robustness(config, df_is, instrument)

            # Walk-Forward
            wf_result = None
            try:
                wf_raw = run_walk_forward(
                    df=df, config=config, instrument=instrument,
                    n_windows=5, is_pct=0.60, overlap=0.5
                )
                wf_windows = []
                for w in wf_raw.get('windows', []):
                    m_oos = w.get('metrics_oos', {})
                    wf_windows.append({
                        'window': w['window'],
                        'start': w['start'][:10],
                        'end': w['end'][:10],
                        'n_bars': w['n_bars'],
                        'oos_trades': m_oos.get('n_trades', 0),
                        'oos_calmar': round(m_oos.get('calmar', 0), 3),
                        'oos_pf': round(m_oos.get('profit_factor', 0), 3),
                        'oos_dd': round(m_oos.get('max_dd', 0), 2),
                        'oos_wr': round(m_oos.get('win_rate', 0), 1),
                    })
                wf_result = {
                    'stability_score': wf_raw.get('stability_score', 0),
                    'consistency': wf_raw.get('consistency', 0),
                    'avg_calmar': wf_raw.get('avg_calmar', 0),
                    'avg_pf': wf_raw.get('avg_pf', 0),
                    'degradation': wf_raw.get('degradation', False),
                    'n_windows': len(wf_windows),
                    'windows': wf_windows,
                }
            except Exception as wf_err:
                wf_result = {'stability_score': 0, 'consistency': 0,
                             'avg_calmar': 0, 'avg_pf': 0,
                             'degradation': False, 'n_windows': 0,
                             'windows': [], 'error': str(wf_err)}

            # Compute robustness grade (0-100)
            rob_score = 0
            # OOS gate pass: 30 pts
            if oos_gate['all_pass']:
                rob_score += 30
            # WFE: up to 20 pts
            rob_score += min(20, max(0, wfe * 20))
            # Perturbation: up to 20 pts
            if perturbation.get('stable', False):
                rob_score += 15
            rob_score += max(0, (1 - perturbation.get('degradation', 1)) * 5)
            # Walk-Forward: up to 30 pts
            if wf_result:
                wf_stab = wf_result.get('stability_score', 0)
                rob_score += min(20, wf_stab / 100 * 20)
                rob_score += min(5, wf_result.get('consistency', 0) / 100 * 5)
                if not wf_result.get('degradation', True):
                    rob_score += 5
            rob_score = round(min(100, max(0, rob_score)), 1)

            if rob_score >= 75:
                grade = 'ROBUST'
            elif rob_score >= 50:
                grade = 'ACCEPTABLE'
            elif rob_score >= 30:
                grade = 'MODERATE'
            else:
                grade = 'FRAGILE'

            # Sanitize for template
            result_data = _sanitize_for_json({
                'config': config,
                'instrument': instrument,
                'is_range': is_range,
                'oos_range': oos_range,
                'metrics_is': {k: v for k, v in metrics_is.items()
                               if k not in ('equity_curve', 'equity_abs', 'drawdown_curve')},
                'metrics_oos': {k: v for k, v in metrics_oos.items()
                                if k not in ('equity_curve', 'equity_abs', 'drawdown_curve')},
                'oos_gate': oos_gate,
                'wfe': wfe,
                'perturbation': perturbation,
                'walk_forward': wf_result,
                'robustness': {'score': rob_score, 'grade': grade},
            })

            return render_template('test.html', result=result_data,
                                   instrument_groups=INSTRUMENT_GROUPS,
                                   session_instrument=instrument)

        except Exception as e:
            import traceback
            return render_template('test.html',
                                   error=f"{str(e)}\n\n{traceback.format_exc()}",
                                   instrument_groups=INSTRUMENT_GROUPS,
                                   session_instrument=instrument,
                                   prefill_config=json.dumps(config, indent=2))

    return render_template('test.html',
                           instrument_groups=INSTRUMENT_GROUPS,
                           session_instrument=session.get('instrument', 'UK100'))


@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio_page():
    """Page 4: Portfolio Analysis (multi-instrument)"""
    if request.method == 'POST':
        files = request.files.getlist('portfolio_files')
        if not files or len(files) < 2:
            return render_template('portfolio.html',
                                 error='Please upload at least 2 trade files.')

        instruments = []
        all_pnls = []
        all_pnls_usd = []
        instrument_stats = []
        parse_errors = []

        for file in files:
            if file.filename:
                filepath = os.path.join(UPLOAD_FOLDER, 'port_' + file.filename)
                file.save(filepath)

                try:
                    pnls_pct, pnls_usd, n_trades, info = load_tv_trades_smart(filepath)

                    # Use pct for correlation/equity, usd for display
                    if len(pnls_pct) > 0:
                        pnls = pnls_pct
                        pnl_type = 'pct'
                    elif len(pnls_usd) > 0:
                        pnls = pnls_usd
                        pnl_type = 'usd'
                    else:
                        parse_errors.append(f"{file.filename}: no P&L data found (sheet: {info['sheet_used']})")
                        continue

                    if len(pnls) < 3:
                        parse_errors.append(f"{file.filename}: only {len(pnls)} trades (need >= 3)")
                        continue

                    # Extract instrument name from filename
                    inst_name = os.path.splitext(file.filename)[0]
                    # Clean up common prefixes
                    for prefix in ['PFA69.1_PEPPERSTONE_', 'PFA69.1_', 'TV_trades_', 'port_']:
                        if inst_name.upper().startswith(prefix.upper()):
                            inst_name = inst_name[len(prefix):]

                    instruments.append(inst_name)
                    all_pnls.append(pnls)
                    all_pnls_usd.append(pnls_usd if len(pnls_usd) > 0 else pnls)

                    # Per-instrument stats
                    total_pnl_usd = round(float(np.sum(pnls_usd)), 2) if len(pnls_usd) > 0 else None
                    total_pnl_pct = round(float(np.sum(pnls_pct)), 2) if len(pnls_pct) > 0 else None
                    instrument_stats.append({
                        'name': inst_name,
                        'n_trades': n_trades,
                        'win_rate': round(float(np.mean(pnls > 0) * 100), 1),
                        'total_pnl_usd': total_pnl_usd,
                        'total_pnl_pct': total_pnl_pct,
                        'max_dd_pct': round(float(np.min(np.minimum.accumulate(np.cumsum(pnls)) - np.maximum.accumulate(np.cumsum(pnls)))), 2) if len(pnls) > 0 else 0,
                        'avg_trade': round(float(np.mean(pnls)), 4),
                        'sheet': info['sheet_used'],
                        'pnl_type': pnl_type,
                    })
                except Exception as e:
                    parse_errors.append(f"{file.filename}: {str(e)}")

        if len(all_pnls) < 2:
            error_msg = 'Need at least 2 valid files with P&L data.'
            if parse_errors:
                error_msg += '\n\nParse errors:\n' + '\n'.join(parse_errors)
            return render_template('portfolio.html', error=error_msg)

        # Correlation matrix
        max_len = max(len(p) for p in all_pnls)
        padded = []
        for p in all_pnls:
            padded_arr = np.zeros(max_len)
            padded_arr[:len(p)] = p
            padded.append(padded_arr)

        corr_matrix = np.corrcoef(padded)

        # Combined equity (equal weight)
        combined_equity = np.zeros(max_len)
        for p in padded:
            combined_equity += np.cumsum(p) / len(all_pnls)

        # Portfolio-level stats
        combined_dd = np.min(combined_equity - np.maximum.accumulate(combined_equity))

        portfolio = {
            'instruments': instruments,
            'n_instruments': len(instruments),
            'correlation_matrix': corr_matrix.tolist(),
            'combined_equity': combined_equity.tolist(),
            'individual_equities': [np.cumsum(p).tolist() for p in padded],
            'instrument_stats': instrument_stats,
            'portfolio_return': round(float(combined_equity[-1]), 2),
            'portfolio_max_dd': round(float(combined_dd), 2),
            'parse_errors': parse_errors,
        }

        return render_template('portfolio.html', portfolio=portfolio)

    return render_template('portfolio.html')


# ==============================================================================
# AUDIT PAGE — V5.4: Quality Audit Dashboard (C6)
# ==============================================================================

@app.route('/api/parse_tv_settings', methods=['POST'])
def api_parse_tv_settings():
    """API: Parse alert text or JSON with config. Supports:
    - Alert text with PFLAB_CONFIG: prefix
    - Raw JSON
    - Webhook JSON payload
    - Multi-line table copy
    """
    text = request.json.get('text', '') if request.is_json else request.form.get('text', '')
    if not text or not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    result = parse_alert_with_config(text)
    return jsonify(result)


@app.route('/api/tv_mapping_guide')
def api_tv_mapping_guide():
    """API: Return the Pine variable -> JSON key mapping guide."""
    guide = generate_tv_mapping_guide()
    return jsonify({'guide': guide, 'count': len(guide)})


@app.route('/api/pine_export_code')
def api_pine_export_code():
    """API: Return the PineScript code block with alert enrichment."""
    code = generate_pine_alert_block()
    instructions = generate_pine_patch_instructions()
    return jsonify({
        'code': code,
        'instructions': instructions,
        'lines': code.count('\n') + 1,
    })


@app.route('/audit')
def audit_page():
    """Page: Quality Audit Dashboard — filterable table with grades, charts."""
    stats = backtest_db.get_stats()
    return render_template('audit.html',
                         stats=stats,
                         instrument_groups=INSTRUMENT_GROUPS)


@app.route('/api/db/audit')
def api_db_audit():
    """API: audit data — all configs with quality grades and robustness scores."""
    from pf_ma_optimizer.metrics import compute_robustness_score, compute_quality_grade
    from pf_ma_optimizer.config import TRANSACTION_COSTS, DEFAULT_TRANSACTION_COST

    instrument = request.args.get('instrument', '')
    grade_filter = request.args.get('grade', '')  # A, B, C, D
    oos_only = request.args.get('oos_gate_only', '0') == '1'
    min_quality = request.args.get('min_quality', 0, type=float)

    configs = backtest_db.get_configs(
        instrument=instrument or None,
        oos_gate_only=oos_only,
        limit=2000)

    from pf_ma_optimizer.config import get_oos_dd_limit

    audit_data = []
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    entry_counts = {}

    for c in configs:
        inst = c.get('instrument', 'UNKNOWN')
        dd_limit = get_oos_dd_limit(inst)
        oos_calmar_val = c.get('oos_calmar') or 0
        oos_pf_val = c.get('oos_profit_factor') or 0
        oos_dd_val = c.get('oos_max_dd') or 0
        oos_trades_val = c.get('oos_trades') or 0

        gate_pass = (
            oos_calmar_val >= 0.25 and
            oos_pf_val >= 1.2 and
            oos_dd_val >= dd_limit and
            oos_trades_val >= 15
        )

        # Perf quality (simplified inline)
        perf_quality = 5.0 if gate_pass else 0.0
        perf_quality += min(max(0, oos_calmar_val), 2.0)

        wfe = c.get('wfe')
        if wfe is not None:
            wfe_capped = min(max(wfe, 0), 1.0)
            perf_quality += wfe_capped * 2.5
            if wfe > 1.5:
                perf_quality -= 1.5
            elif wfe > 1.2:
                perf_quality -= 0.8
            elif wfe > 1.0:
                perf_quality -= 0.3

        wf_stab = c.get('wf_stability') or 0
        wf_cons = c.get('wf_consistency') or 0
        perf_quality += (min(wf_stab, 100) / 100.0) * 2.0
        perf_quality += (min(wf_cons, 100) / 100.0) * 0.5
        if c.get('perturbation_stable'):
            perf_quality += 0.5
        perf_quality += (c.get('score') or 0) * 0.2

        robustness = compute_robustness_score(
            wfe=wfe,
            perturbation_stable=bool(c.get('perturbation_stable')),
            perturbation_degradation=c.get('perturbation_degradation'),
            wf_stability=c.get('wf_stability'),
            wf_consistency=c.get('wf_consistency'),
            wf_degradation=bool(c.get('wf_degradation')),
        )

        grade_info = compute_quality_grade(
            perf_score=perf_quality,
            robustness_score=robustness,
            entry_type=c.get('entry_type', ''),
            instrument=inst,
            transaction_costs=TRANSACTION_COSTS,
            default_cost=DEFAULT_TRANSACTION_COST,
        )

        grade = grade_info['grade']
        quality = grade_info['quality_score']

        if grade_filter and grade != grade_filter:
            continue
        if quality < min_quality:
            continue

        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        et = c.get('entry_type', 'Unknown')
        entry_counts[et] = entry_counts.get(et, 0) + 1

        c.pop('config_json', None)
        c.pop('tv_export', None)
        c.pop('oos_trades_json', None)

        c['_grade'] = grade
        c['_quality'] = quality
        c['_robustness'] = grade_info['robustness_score']
        c['_flags'] = grade_info['flags']
        c['_perf_quality'] = round(perf_quality, 4)
        c['oos_gate_pass'] = int(gate_pass)
        audit_data.append(c)

    audit_data.sort(key=lambda x: x['_quality'], reverse=True)

    return jsonify({
        'configs': audit_data[:500],
        'total': len(audit_data),
        'grade_counts': grade_counts,
        'entry_counts': entry_counts,
    })


if __name__ == '__main__':
    import sys
    # debug=False by default — Flask reloader causes issues on Windows
    use_debug = '--debug' in sys.argv
    port = 5000
    if '--port' in sys.argv:
        idx = sys.argv.index('--port')
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    # ── Windows connection-refused fix ─────────────────────────────
    # On Windows, host='0.0.0.0' triggers firewall popups and the
    # default single-threaded Werkzeug server blocks when the browser
    # opens multiple connections (page + favicon + CSS + JS).
    # Fix: use 127.0.0.1 on Windows + always enable threaded mode.
    host = '127.0.0.1' if sys.platform == 'win32' else '0.0.0.0'
    app.run(host=host, port=port, debug=use_debug, threaded=True)
