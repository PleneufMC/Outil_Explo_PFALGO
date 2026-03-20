# PF AI Lab -- Analyse & Propositions d'Ameliorations

**Date** : 9 mars 2026
**Version analysee** : PF AI Lab 5.5 / P.F.Algo V.69.4
**Base de donnees** : 40 runs, 800 configs, 9 instruments, 49 023 trades OOS

---

## Section A -- Ameliorations du PF AI Lab

---

### A1. Optimisation des metriques

#### A1.1 Omega Ratio -- PRIORITE HAUTE

**Description** : Le Omega Ratio (rendements au-dessus d'un seuil / rendements en dessous) capture toute la distribution des returns, pas seulement les deux premiers moments. Contrairement au Sharpe qui suppose une distribution normale, Omega est sensible a la skewness et au kurtosis -- crucial pour le trading.

**Impact estime** : Eleve. Permet de detecter des strategies dont la queue gauche est dangereuse malgre un bon Sharpe.

**Effort estime** : Faible (0.5 jour). Le calcul est simple : `sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))` sur les returns quotidiens.

**Exemple concret** :
```python
# Dans metrics.py
def omega_ratio(daily_returns: pd.Series, threshold: float = 0.0) -> float:
    excess = daily_returns - threshold
    return float(excess[excess > 0].sum() / abs(excess[excess < 0].sum())) if excess[excess < 0].sum() != 0 else float('inf')
```

#### A1.2 Tail Ratio -- PRIORITE HAUTE

**Description** : Le Tail Ratio mesure la symetrie des queues de distribution : `percentile_95 / abs(percentile_5)`. Un ratio > 1 indique que les gros gains sont plus importants que les grosses pertes -- signe de robustesse.

**Impact estime** : Eleve. Identifie les strategies qui "explosent" en drawdown malgre de bonnes stats moyennes. Actuellement, seul le Max DD capture cet aspect, mais il ne dit rien sur la frequence des pertes extremes.

**Effort estime** : Faible (0.25 jour).

**Exemple concret** :
```python
def tail_ratio(daily_returns: pd.Series) -> float:
    p95 = np.percentile(daily_returns, 95)
    p5 = np.percentile(daily_returns, 5)
    return float(p95 / abs(p5)) if p5 != 0 else float('inf')
```

#### A1.3 Profit Factor par regime de marche -- PRIORITE MOYENNE

**Description** : Calculer le Profit Factor separement pour les periodes trending vs ranging (via ADX ou volatilite ATR). Actuellement, le PF global masque les periodes ou la strategie perd systematiquement.

**Impact estime** : Moyen-Eleve. Permet de savoir si une strategie est dependante d'un regime specifique.

**Effort estime** : Moyen (1-2 jours). Necessite de segmenter les trades par regime, ce qui implique de calculer l'ADX sur la periode OOS.

**Exemple concret** :
```python
def pf_by_regime(trades, ohlc_data, adx_threshold=25):
    adx = compute_adx(ohlc_data, period=14)
    trending_trades = [t for t in trades if adx[t['entry_bar']] > adx_threshold]
    ranging_trades = [t for t in trades if adx[t['entry_bar']] <= adx_threshold]
    return {
        'trending_pf': profit_factor(trending_trades),
        'ranging_pf': profit_factor(ranging_trades),
        'trending_pct': len(trending_trades) / len(trades) * 100
    }
```

#### A1.4 Expectancy per trade en unites de risque -- PRIORITE MOYENNE

**Description** : L'expectancy actuelle (`oos_expectancy`) est en pourcentage brut. La convertir en multiples de risque initial (R-multiples) permettrait une comparaison normalisee entre instruments et configurations SL differentes.

**Impact estime** : Moyen. Meilleure comparabilite cross-instrument.

**Effort estime** : Faible (0.5 jour). Chaque trade a un `sl_pct` connu via la config.

**Exemple concret** :
```python
def expectancy_R(trades, sl_pct):
    """Expectancy en multiples du risque initial."""
    r_multiples = [t['pnl'] / sl_pct for t in trades]
    return np.mean(r_multiples)
```

#### A1.5 Ulcer Index -- PRIORITE BASSE

**Description** : Le Ulcer Index mesure la profondeur ET la duree des drawdowns (racine de la moyenne des carres des drawdowns). Plus sensible que le Max DD simple car il penalise les drawdowns prolonges.

**Impact estime** : Moyen. Complementaire au Max DD pour detecter les periodes de sous-performance longues.

**Effort estime** : Faible (0.25 jour).

#### A1.6 Metriques de distribution des trades -- PRIORITE MOYENNE

**Description** : Ajouter skewness et kurtosis de la distribution des PnL par trade. Une strategie avec un kurtosis eleve et une skewness negative est dangereuse meme si le PF est bon.

**Impact estime** : Moyen. Aide a filtrer les strategies a risque de tail events.

**Effort estime** : Faible (0.5 jour). Scipy `skew()` et `kurtosis()` sur les PnL trades.

---

### A2. Walk-Forward Analysis

#### A2.1 Anchored Walk-Forward -- PRIORITE HAUTE

**Description** : Le WFA actuel utilise 5 fenetres glissantes de taille fixe (60/40, 50% overlap). Ajouter un mode "anchored" ou la fenetre IS commence toujours au meme point et s'agrandit progressivement. L'anchored WF est plus realiste car il simule ce que ferait un trader qui accumule des donnees au fil du temps.

**Impact estime** : Eleve. Le rolling WF peut masquer une degradation structurelle si les premieres fenetres sont bonnes. L'anchored WF detecte mieux la degradation temporelle.

**Effort estime** : Moyen (2 jours). Modifier `walk_forward.py` pour supporter les deux modes.

**Exemple concret** :
```python
# Mode anchored : IS toujours depuis le debut, OOS avance
# Window 1: IS=[0, 60%], OOS=[60%, 72%]
# Window 2: IS=[0, 72%], OOS=[72%, 84%]
# Window 3: IS=[0, 84%], OOS=[84%, 96%]
# Window 4: IS=[0, 96%], OOS=[96%, 100%]

def anchored_walk_forward(data, config, n_windows=4, first_oos_ratio=0.4):
    total = len(data)
    oos_size = int(total * first_oos_ratio / n_windows)
    results = []
    for w in range(n_windows):
        oos_end = total - (n_windows - 1 - w) * oos_size
        oos_start = oos_end - oos_size
        is_data = data[:oos_start]
        oos_data = data[oos_start:oos_end]
        # Re-optimize on IS, test on OOS
        ...
```

#### A2.2 Fenetres OOS multiples avec recalibration -- PRIORITE MOYENNE

**Description** : Actuellement le WF teste la MEME config sur 5 fenetres. Un vrai WF devrait re-optimiser sur chaque fenetre IS puis tester sur l'OOS suivante. C'est beaucoup plus couteux mais nettement plus fiable.

**Impact estime** : Tres eleve. C'est la difference entre un "pseudo-WF" et un vrai Walk-Forward selon Pardo.

**Effort estime** : Eleve (5 jours). Necessite de lancer une mini-optimisation par fenetre (50-100 trials), ce qui multiplie le temps de calcul par 5.

**Exemple concret** : Ajouter un mode `wf_mode='full_reoptimize'` dans `optimizer.py` qui lance une boucle :
1. Optimiser sur IS window k (50 trials)
2. Tester le top-1 sur OOS window k
3. Collecter les metriques OOS
4. Passer a la fenetre k+1

#### A2.3 Metriques WF supplementaires -- PRIORITE MOYENNE

**Description** : Ajouter au rapport Walk-Forward :
- **WF Sharpe** : Sharpe calcule sur la serie des returns OOS concatenes
- **WF Max DD** : Max DD sur la serie OOS concatenee
- **WF Equity Curve** : la courbe d'equity OOS-only (stockee dans la DB pour visualisation)

**Impact estime** : Moyen. Donne une vue plus complete de la performance "live simulee".

**Effort estime** : Faible-Moyen (1 jour).

#### A2.4 WF avec Purge et Embargo -- PRIORITE BASSE

**Description** : Ajouter un gap (purge/embargo) entre IS et OOS pour eviter le data leaking temporel. Typiquement 5-10% de la fenetre IS est supprime a la jonction IS/OOS. Cela evite que les dernieres barres IS contaminent les signaux OOS.

**Impact estime** : Faible-Moyen. Surtout pertinent pour les signaux RSI Divergence et les MA longues.

**Effort estime** : Faible (0.5 jour).

---

### A3. Filtrage et Scoring

#### A3.1 OOS Gate multi-niveaux -- PRIORITE HAUTE

**Description** : Le gate actuel est binaire (PASS/FAIL sur 4 criteres). Proposer un scoring continu :
- **GOLD** : Calmar OOS >= 1.0 ET PF >= 1.5 ET trades >= 40 ET perturbation stable ET WFE >= 0.5
- **SILVER** : Calmar OOS >= 0.50 ET PF >= 1.30 ET trades >= 25
- **BRONZE** : Calmar OOS >= 0.25 ET PF >= 1.20 ET trades >= 15 (actuel)
- **FAIL** : en dessous

**Impact estime** : Eleve. Actuellement, 145/800 configs passent le gate (18.1%). Mais ce 18.1% melange des configs marginales et des configs excellentes. Un tier GOLD permettrait d'identifier immediatement les 10-20 meilleures.

**Effort estime** : Faible (0.5 jour). Modifier `backtest_db.py` et les templates.

**Exemple concret** :
```python
def oos_gate_tier(c):
    if c['oos_calmar'] >= 1.0 and c['oos_profit_factor'] >= 1.5 and c['oos_trades'] >= 40 and c['perturbation_stable'] and c['wfe'] >= 0.5:
        return 'GOLD'
    elif c['oos_calmar'] >= 0.50 and c['oos_profit_factor'] >= 1.30 and c['oos_trades'] >= 25:
        return 'SILVER'
    elif c['oos_calmar'] >= 0.25 and c['oos_profit_factor'] >= 1.20 and c['oos_trades'] >= 15:
        return 'BRONZE'
    return 'FAIL'
```

#### A3.2 Coherence IS/OOS obligatoire dans le gate -- PRIORITE HAUTE

**Description** : Les flags anti-overfit v5.5 (`OVERFIT_PF_RATIO`, `IS_PF_TOO_HIGH`, `OVERFIT_WR_DELTA`) sont informatifs mais ne bloquent pas le gate. Les integrer comme conditions eliminatoires :
- PF Ratio IS/OOS > 2.5 => FAIL automatique
- Delta Win Rate > 15pp => FAIL automatique

**Impact estime** : Moyen. Elimine les faux positifs qui passent le gate grace a un OOS chanceux.

**Effort estime** : Faible (0.25 jour).

#### A3.3 Score de diversification -- PRIORITE BASSE

**Description** : Dans le Quality Score V2, ajouter un bonus si la strategie fonctionne sur des exit reasons variees (pas 100% `setup_revers`). Actuellement, les donnees montrent que 85.6% des trades sortent par `setup_revers`, ce qui indique une faible diversification des sorties.

**Impact estime** : Faible. Plus informatif qu'eliminatoire.

**Effort estime** : Faible (0.5 jour).

---

### A4. Perturbation Testing

#### A4.1 Perturbation asymetrique -- PRIORITE HAUTE

**Description** : Le test actuel perturbe de +/-10% sur 5 runs. Ameliorations :
- **Perturbation graduee** : 5%, 10%, 15%, 20% pour construire une "courbe de sensibilite"
- **Perturbation directionnelle** : tester separement +10% et -10% (certains parametres ne sont sensibles que dans une direction)
- **Augmenter a 10-20 runs** par niveau pour une meilleure significativite

**Impact estime** : Eleve. 5 runs a +/-10% est un echantillon trop faible pour etre statistiquement fiable. 20 runs avec 4 niveaux donnerait 80 backtests supplementaires mais une vision beaucoup plus complete.

**Effort estime** : Moyen (1-2 jours). Paralleliser avec ProcessPoolExecutor existant.

**Exemple concret** :
```python
perturbation_levels = [0.05, 0.10, 0.15, 0.20]
results = {}
for level in perturbation_levels:
    runs = [perturb_and_backtest(config, level) for _ in range(20)]
    results[level] = {
        'survival_rate': sum(1 for r in runs if r['calmar'] > 0.5 * base_calmar) / len(runs),
        'avg_degradation': 1 - np.mean([r['calmar'] for r in runs]) / base_calmar
    }
# Courbe de sensibilite : survival_rate en fonction du level
```

#### A4.2 Perturbation des donnees (data perturbation) -- PRIORITE MOYENNE

**Description** : En plus de perturber les parametres, perturber les DONNEES : ajouter du bruit gaussien sur les prix (0.1-0.5% du prix), decaler les barres de 1-2 positions, supprimer aleatoirement 5% des barres. Cela teste la robustesse au bruit de marche et au slippage.

**Impact estime** : Eleve. Une strategie qui survit a la perturbation des parametres mais pas des donnees est fragile en conditions reelles.

**Effort estime** : Moyen (2 jours).

**Exemple concret** :
```python
def perturb_data(ohlc, noise_pct=0.002, drop_pct=0.05):
    """Ajoute du bruit et supprime des barres aleatoirement."""
    noisy = ohlc.copy()
    noise = np.random.normal(0, noise_pct, size=len(ohlc))
    for col in ['Open', 'High', 'Low', 'Close']:
        noisy[col] = ohlc[col] * (1 + noise * np.random.randn(len(ohlc)))
    # Drop random bars
    drop_idx = np.random.choice(len(noisy), int(len(noisy) * drop_pct), replace=False)
    noisy = noisy.drop(noisy.index[drop_idx]).reset_index(drop=True)
    return noisy
```

#### A4.3 Perturbation des couts de transaction -- PRIORITE MOYENNE

**Description** : Tester la strategie avec des couts de transaction x1.5 et x2.0. Beaucoup de strategies marginales deviennent non-rentables avec un spread legerement plus large (conditions reelles).

**Impact estime** : Moyen-Eleve. Directement lie a la viabilite en live.

**Effort estime** : Faible (0.5 jour).

---

### A5. Data Enrichment

#### A5.1 Capturer le regime de marche par trade -- PRIORITE HAUTE

**Description** : Pour chaque trade dans `oos_trades_json`, ajouter le regime de marche a l'entree :
- `adx` : valeur ADX au moment de l'entree
- `atr_pctile` : percentile ATR (volatilite)
- `trend_dir` : direction du filtre LT au moment du trade

Cela permettrait une analyse post-hoc par regime sans recalculer les indicateurs.

**Impact estime** : Eleve. Actuellement, les 49 023 trades OOS n'ont que 6 champs (`s, ep, xp, pnl, dur, xr`). Enrichir les trades permettrait toutes les analyses de la section B.

**Effort estime** : Moyen (1-2 jours). Modifier `backtest_engine.py` pour inclure ces champs.

**Exemple concret** (format enrichi d'un trade) :
```json
{
  "s": "l", "ep": 1.11822, "xp": 1.11724, "pnl": -0.101, "dur": 149,
  "xr": "setup_revers",
  "adx": 22.5, "atr_pctile": 45, "lt_dir": "bull",
  "hour": 14, "dow": 2, "bar_idx": 1523,
  "entry_date": "2025-06-15T14:00:00",
  "exit_date": "2025-06-18T07:25:00"
}
```

#### A5.2 Capturer les timestamps des trades -- PRIORITE HAUTE

**Description** : Actuellement, les trades ont une `dur` (duree en barres) mais pas de dates d'entree/sortie. Sans dates, impossible de faire du clustering temporel ou de detecter si les pertes sont concentrees sur certaines periodes.

**Impact estime** : Eleve. C'est un prerequis pour la detection de regime et l'analyse temporelle.

**Effort estime** : Faible (0.5 jour). L'index datetime est disponible dans le backtest engine.

#### A5.3 Stocker les IS trades -- PRIORITE MOYENNE

**Description** : Actuellement, seuls les OOS trades sont stockes (`oos_trades_json`). Il n'y a pas de `is_trades_json`. Stocker aussi les IS trades permettrait de comparer directement les distributions IS vs OOS et de detecter l'overfitting au niveau trade.

**Impact estime** : Moyen. Utilise avec la Section B5 (feature engineering IS/OOS).

**Effort estime** : Faible (0.25 jour). Le champ existe probablement deja dans le backtest engine.

**Attention** : impact sur la taille de la DB. Avec 800 configs x ~100 trades IS en moyenne, ca represente ~80 000 trades supplementaires. Compresser le JSON ou le stocker dans une table separee.

#### A5.4 Capturer les Walk-Forward trades detailles -- PRIORITE BASSE

**Description** : Actuellement, le WF ne sauvegarde que des metriques agregees (stability, consistency, avg_calmar, avg_pf). Stocker les metriques PAR fenetre (calmar_w1, pf_w1, trades_w1, etc.) et les equity curves OOS.

**Impact estime** : Moyen. Permet de visualiser la degradation fenetre par fenetre.

**Effort estime** : Moyen (1 jour).

#### A5.5 Equity curve OOS complete -- PRIORITE MOYENNE

**Description** : Stocker la serie temporelle de l'equity curve OOS (pas seulement les trades). Cela permettrait de calculer le Ulcer Index, l'Omega ratio, et de visualiser le comportement jour par jour.

**Impact estime** : Moyen-Eleve.

**Effort estime** : Moyen (1 jour). Le backtest engine calcule deja l'equity bar-by-bar, il suffit de l'exposer.

---

### A6. Architecture et Performance

#### A6.1 Vectorisation du backtest engine -- PRIORITE HAUTE

**Description** : Le backtest engine utilise une boucle Python barre par barre (inherent au modele 3 phases). Optimiser via :
- Pre-calculer TOUS les indicateurs et signaux en vectoriel (numpy/pandas) AVANT la boucle
- N'utiliser la boucle que pour la logique d'execution (SL/TP/position management)
- Utiliser `numba.jit` pour la boucle d'execution

**Impact estime** : Eleve. Actuellement, 500 trials sur EURUSD prennent ~300s. Un facteur x5-10 est realiste avec numba, ce qui permettrait 5000 trials en temps raisonnable.

**Effort estime** : Eleve (5-7 jours). La boucle d'execution est complexe (SL/TP/BE/TSL/partial TP).

**Exemple concret** :
```python
from numba import njit

@njit
def _execute_loop(opens, highs, lows, closes, signals, filters,
                  sl_prices, tp_prices, position, equity):
    """Boucle d'execution JIT-compiled."""
    for i in range(1, len(opens)):
        if position[i-1] != 0:
            # Phase 2: check SL/TP
            ...
        if signals[i] and filters[i] and position[i-1] == 0:
            # Phase 3: entry
            ...
    return equity, trades
```

#### A6.2 Cache des indicateurs entre trials -- PRIORITE HAUTE

**Description** : Beaucoup de trials partagent les memes indicateurs de base (ATR, ADX, RSI) qui ne dependent pas des parametres optimises. Calculer ces indicateurs une seule fois et les mettre en cache.

**Impact estime** : Moyen. Reduction de 20-30% du temps par trial.

**Effort estime** : Faible-Moyen (1 jour).

#### A6.3 Parallelisation des trials Optuna -- PRIORITE MOYENNE

**Description** : Le post-traitement (WF, perturbation, OOS) des top-20 configs est deja parallelise. Mais les trials Optuna eux-memes sont sequentiels. Utiliser `n_jobs > 1` dans Optuna avec un storage in-memory.

**Impact estime** : Moyen. Speedup proportionnel au nombre de cores.

**Effort estime** : Faible (0.5 jour). `study.optimize(objective, n_trials=500, n_jobs=4)`.

**Attention** : NSGA-II multi-objectif ne supporte pas bien le parallelisme. Verifier avec TPE sampler.

#### A6.4 Database migration vers DuckDB -- PRIORITE BASSE

**Description** : SQLite est suffisant pour le volume actuel (800 configs). Mais si le volume croit (10 000+ configs), les requetes analytiques (aggregation par instrument, cross-run analysis) beneficieraient de DuckDB, qui est colonnaire et optimise pour l'analytique.

**Impact estime** : Faible actuellement, eleve a long terme.

**Effort estime** : Moyen (2-3 jours).

---

### A7. Multi-instrument

#### A7.1 Cross-instrument correlation native -- PRIORITE HAUTE

**Description** : La page Portfolio accepte les trades TradingView, mais pas les trades de la DB. Ajouter un endpoint `/api/portfolio/from_db` qui prend N config IDs et calcule :
- Matrice de correlation des equity curves OOS
- Combined Sharpe et Calmar (portfolio equal-weight)
- Diversification ratio

**Impact estime** : Eleve. Actuellement, il faut exporter les trades et les re-uploader. Avec la DB riche (49 023 trades OOS), tout est deja la.

**Effort estime** : Moyen (2-3 jours).

**Exemple concret** :
```python
@app.route('/api/portfolio/from_db', methods=['POST'])
def portfolio_from_db():
    config_ids = request.json['config_ids']
    configs = [db.get_config(cid) for cid in config_ids]
    equity_curves = [compute_equity_from_trades(json.loads(c['oos_trades_json'])) for c in configs]
    corr_matrix = pd.DataFrame(equity_curves).T.corr()
    combined = sum(equity_curves) / len(equity_curves)
    return jsonify({
        'correlation': corr_matrix.to_dict(),
        'combined_sharpe': compute_sharpe(combined),
        'combined_calmar': compute_calmar(combined),
    })
```

#### A7.2 Universe screening automatique -- PRIORITE MOYENNE

**Description** : Lancer une optimisation legere (100 trials) sur N instruments simultanement, puis classer les instruments par "potentiel" (meilleur score OOS atteignable). Utile pour identifier rapidement quels instruments meritent une exploration approfondie.

**Impact estime** : Moyen. Gain de temps significatif pour l'utilisateur qui explore beaucoup d'instruments.

**Effort estime** : Moyen (2 jours).

#### A7.3 Template configs cross-instrument -- PRIORITE BASSE

**Description** : Quand une config Grade A est trouvee sur un instrument, la tester automatiquement sur les instruments correles (ex: une bonne config EURUSD testee sur GBPUSD, EURGBP). Cela valide la robustesse structurelle du signal.

**Impact estime** : Moyen. Un signal qui marche sur 3 paires Forex est bien plus fiable qu'un signal specifique a une paire.

**Effort estime** : Faible (1 jour). Boucle simple sur les instruments du meme groupe.

---

## Section B -- Meilleure exploitation des donnees existantes

---

### B1. Champs sous-exploites

#### B1.1 `oos_avg_duration` -- QUICK WIN

**Description** : Ce champ existe (valeur mediane ~155 barres dans les PASS configs) mais n'est utilise ni dans le gate, ni dans le scoring, ni dans l'analyse FTMO. La duree moyenne des trades est pourtant critique pour FTMO : des trades tres longs (>100 barres en M5 = 8h+) augmentent l'exposition au overnight risk et au weekend risk.

**Action** : Ajouter un filtre optionnel `max_avg_duration` dans le gate (ex: < 200 barres pour M5).

**Effort** : 0.25 jour.

#### B1.2 `oos_avg_win` / `oos_avg_loss` ratio -- QUICK WIN

**Description** : Le ratio `avg_win / abs(avg_loss)` est stocke mais jamais exploite. C'est le Reward/Risk ratio reel. Un RR < 1.0 avec un Win Rate < 50% est mathematiquement perdant. Ce check devrait etre dans le gate.

**Action** : Calculer `RR = oos_avg_win / abs(oos_avg_loss)` et ajouter une regle : si `WR < 50% AND RR < 1.0 => FAIL`.

**Effort** : 0.25 jour.

#### B1.3 `oos_expectancy` -- QUICK WIN

**Description** : L'expectancy OOS est stockee mais n'apparait ni dans le gate ni dans le Quality Score. C'est pourtant la metrique la plus fondamentale du trading : l'esperance de gain par trade.

**Action** : Ajouter comme condition supplementaire du gate : `oos_expectancy > 0.02` (au moins 0.02% par trade apres couts).

**Effort** : 0.25 jour.

#### B1.4 `config_json` (90 cles) vs `configs` (30 cles exposees) -- PROJET

**Description** : Le `config_json` contient 90 parametres complets mais la table `configs` n'en expose que ~30 en colonnes. Les 60 parametres caches incluent :
- `max_hold_days`, `tp_mode`, `be_mode`, `tsl_pct` (gestion du risque avancee)
- `use_session_filter`, `session_allowed_days` (timing)
- `use_adx_regime_filter`, `use_vol_filter` (filtres de regime)
- `mm_cross_ma1_type/len`, `mm_cross_ma2_type/len` (parametres MM Cross)

**Action** : Extraire les champs les plus importants en colonnes DB pour permettre des filtres SQL directs :
```sql
SELECT * FROM configs WHERE max_hold_days > 0 AND tp_mode != 'no'
```

**Effort** : 1-2 jours (migration DB + ajout colonnes).

#### B1.5 `long_side` / `short_side` asymetrie -- QUICK WIN

**Description** : Dans la DB, les donnees montrent un biais massif : 40 129 trades long vs 8 894 trades short (82% long). Cela pourrait indiquer que les strategies short sont sous-optimisees ou que les filtres LT empechent les shorts. Ce biais devrait etre visible dans l'UI.

**Action** : Ajouter un indicateur "Long/Short Ratio" dans la page Best Configs et un avertissement si > 90% d'un cote.

**Effort** : 0.25 jour.

---

### B2. Exploitation de `oos_trades_json`

#### B2.1 Distribution des PnL et detection d'outliers -- QUICK WIN

**Description** : Les 49 023 trades OOS ont des PnL allant de -11.0% a +33.3%. Calculer pour chaque config :
- Skewness et kurtosis de la distribution PnL
- Nombre de trades > 2 ecarts-types (outliers)
- Pourcentage du PnL total apporte par les top 5 trades

**Action** : Endpoint `/api/analysis/pnl_distribution/<config_id>` qui retourne ces stats.

**Effort** : 0.5 jour.

**Exemple concret** :
```python
def analyze_pnl_distribution(trades_json):
    pnls = [t['pnl'] for t in json.loads(trades_json)]
    return {
        'mean': np.mean(pnls),
        'std': np.std(pnls),
        'skew': scipy.stats.skew(pnls),
        'kurtosis': scipy.stats.kurtosis(pnls),
        'outliers_2std': sum(1 for p in pnls if abs(p - np.mean(pnls)) > 2 * np.std(pnls)),
        'top5_contribution': sum(sorted(pnls)[-5:]) / max(sum(p for p in pnls if p > 0), 0.001),
        'worst5_contribution': sum(sorted(pnls)[:5]) / min(sum(p for p in pnls if p < 0), -0.001),
    }
```

#### B2.2 Autocorrelation des trades -- QUICK WIN

**Description** : Tester si les trades gagnants/perdants arrivent en clusters (autocorrelation de lag-1 a lag-5 de la serie win/loss). Une forte autocorrelation positive signifie que les pertes arrivent en rafales -- dangereux pour FTMO.

**Action** :
```python
def trade_autocorrelation(trades_json, max_lag=5):
    outcomes = [1 if t['pnl'] > 0 else 0 for t in json.loads(trades_json)]
    series = pd.Series(outcomes)
    return {f'lag_{i}': float(series.autocorr(lag=i)) for i in range(1, max_lag + 1)}
```

**Effort** : 0.25 jour.

#### B2.3 Clustering temporel des pertes -- PROJET

**Description** : Avec les timestamps des trades (cf A5.2), identifier les periodes de drawdown prolonge. Questions a repondre :
- Les pertes sont-elles concentrees sur certains jours de la semaine ?
- Y a-t-il une saisonnalite intraday ? (trades perdants a l'ouverture US vs session asiatique)
- Le drawdown max est-il un evenement unique ou repete ?

**Prerequis** : A5.2 (timestamps dans les trades).

**Effort** : 1-2 jours apres A5.2.

#### B2.4 Analyse par exit reason -- QUICK WIN

**Description** : Les donnees montrent 3 exit reasons : `setup_revers` (85.6%), `max_hold_day` (14.0%), `end_of_data` (0.4%). Calculer le PF et le win rate PAR exit reason. Si `max_hold_day` est systematiquement perdant, le `max_hold_days` est mal calibre.

**Action** :
```python
def metrics_by_exit_reason(trades_json):
    trades = json.loads(trades_json)
    by_reason = {}
    for xr in set(t['xr'] for t in trades):
        subset = [t for t in trades if t['xr'] == xr]
        wins = [t['pnl'] for t in subset if t['pnl'] > 0]
        losses = [t['pnl'] for t in subset if t['pnl'] <= 0]
        by_reason[xr] = {
            'count': len(subset),
            'win_rate': len(wins) / len(subset) * 100 if subset else 0,
            'avg_pnl': np.mean([t['pnl'] for t in subset]),
            'total_pnl': sum(t['pnl'] for t in subset),
        }
    return by_reason
```

**Effort** : 0.25 jour.

#### B2.5 Run-length analysis (series gagnantes/perdantes) -- QUICK WIN

**Description** : Calculer la longueur maximale et moyenne des series de trades gagnants et perdants consecutifs. FTMO exige de passer les challenges rapidement -- une serie de 10 pertes consecutives peut etre fatale.

**Effort** : 0.25 jour.

---

### B3. Cross-Run Analysis

#### B3.1 Stabilite parametrique cross-runs -- PROJET

**Description** : La DB contient 40 runs sur 9 instruments. Pour chaque instrument, identifier les parametres qui apparaissent systematiquement dans les PASS configs :
- PMax length converge-t-il vers une valeur ?
- Le filtre LT est-il toujours active ?
- Quel entry_type domine ?

**Action** : Creer un rapport "Parameter Convergence" par instrument :
```python
def parameter_convergence(instrument):
    pass_configs = db.get_configs(instrument=instrument, oos_gate_only=True)
    report = {}
    for param in ['pmax_length', 'pmax_multiplier', 'entry_type', 'use_lt_filter', 'sl_pct']:
        values = [c[param] for c in pass_configs]
        if isinstance(values[0], (int, float)):
            report[param] = {'mean': np.mean(values), 'std': np.std(values), 'mode': scipy.stats.mode(values)[0]}
        else:
            report[param] = dict(Counter(values))
    return report
```

**Effort** : 1 jour.

#### B3.2 Meta-learning : quels parametres predisent le OOS success -- PROJET

**Description** : Entrainer un modele simple (logistic regression, random forest) sur les 800 configs pour predire `oos_gate_pass` a partir des parametres IS + metriques IS. Cela identifierait les "predicteurs d'overfitting".

**Features** : `is_calmar`, `is_profit_factor`, `is_win_rate`, `is_trades`, `pmax_length`, `pmax_multiplier`, `entry_type`, `use_lt_filter`, `use_mt_filter`, ...

**Target** : `oos_gate_pass`

**Impact estime** : Eleve. Si le modele identifie que `is_calmar > 3.0` predit systematiquement `oos_gate_pass = 0`, c'est un signal d'overfit puissant.

**Effort** : 2 jours.

#### B3.3 Run similarity et clustering -- PROJET

**Description** : Calculer la distance entre les configs (cosine similarity sur les parametres normalises) pour identifier les "familles" de strategies. Si 5 configs PASS sont tres similaires, elles ne comptent que pour une seule strategie en termes de diversification.

**Effort** : 1-2 jours.

---

### B4. Regime Detection

#### B4.1 Proxy regime via volatilite des trades -- QUICK WIN

**Description** : Sans donnees OHLCV (non stockees en DB), on peut inferer le regime a partir des trades :
- Volatilite des PnL par fenetre glissante de 20 trades
- Duree moyenne des trades par fenetre (trades courts = marche rapide)
- Ratio trades gagnants/perdants par fenetre

Quand la volatilite des PnL augmente et le win rate chute, c'est un changement de regime.

**Effort** : 0.5 jour.

#### B4.2 Detection de breakpoints dans l'equity curve -- PROJET

**Description** : Utiliser un algorithme de detection de changements structurels (CUSUM, Bai-Perron) sur l'equity curve OOS pour identifier les moments ou la strategie "casse". Cela donnerait une date de peremption estimee.

**Prerequis** : A5.5 (equity curve stockee).

**Effort** : 1-2 jours.

#### B4.3 Regime-conditional performance -- PROJET

**Description** : En combinant B4.1 et A5.1, construire un rapport "Performance par regime" :

| Regime | Trades | Win Rate | PF | Avg PnL |
|--------|--------|----------|-----|---------|
| Trending | 45 | 42% | 1.8 | +0.15% |
| Ranging | 23 | 22% | 0.6 | -0.05% |
| High Vol | 12 | 33% | 0.9 | -0.02% |

**Prerequis** : A5.1 (regime dans les trades).

**Effort** : 1 jour apres A5.1.

---

### B5. Feature Engineering

#### B5.1 IS/OOS degradation vector -- QUICK WIN

**Description** : Pour chaque config, calculer un vecteur de degradation sur les metriques cles :
```python
degradation = {
    'calmar_deg': (is_calmar - oos_calmar) / max(is_calmar, 0.01),
    'pf_deg': (is_pf - oos_pf) / max(is_pf, 0.01),
    'wr_deg': (is_wr - oos_wr) / max(is_wr, 0.01),
    'trades_ratio': oos_trades / max(is_trades, 1),
}
```

Visualiser ce vecteur sur un radar chart pour chaque config. Une degradation uniforme (tout baisse de ~30%) est saine. Une degradation asymetrique (Calmar chute de 80% mais WR stable) indique un probleme specifique.

**Effort** : 0.5 jour.

#### B5.2 Score FTMO-readiness -- QUICK WIN

**Description** : Creer un score specifique pour la compatibilite FTMO a partir des donnees existantes :
- **Max DD OOS** < 5% (challenge FTMO : max 10% daily, 5% max trailing)
- **Series perdante max** < 5 trades consecutifs
- **Profit target** : oos_net_return > 10% (objectif FTMO 10% sur 30 jours)
- **Trades par jour** : oos_trades / (nombre_jours_OOS) > 0.5 (au moins 1 trade tous les 2 jours)
- **Expectancy** > 0.1% par trade

```python
def ftmo_readiness(config, oos_days):
    trades = json.loads(config['oos_trades_json'])
    max_consec_loss = max_consecutive_losses(trades)
    trades_per_day = len(trades) / oos_days
    return {
        'dd_ok': config['oos_max_dd'] >= -5.0,
        'consec_loss_ok': max_consec_loss <= 5,
        'profit_ok': config['oos_net_return'] >= 10.0,
        'frequency_ok': trades_per_day >= 0.5,
        'expectancy_ok': config['oos_expectancy'] > 0.1,
        'ftmo_score': sum([...]) / 5 * 100
    }
```

**Effort** : 0.5 jour.

#### B5.3 Stability decay index -- PROJET

**Description** : Diviser les trades OOS en 4 quartiles chronologiques et calculer les metriques par quartile. Si la performance decline du Q1 au Q4, la strategie se degrade dans le temps.

```python
def stability_decay(trades_json):
    trades = json.loads(trades_json)
    n = len(trades)
    quartiles = [trades[i*n//4:(i+1)*n//4] for i in range(4)]
    return {
        f'Q{i+1}_winrate': sum(1 for t in q if t['pnl'] > 0) / len(q) * 100,
        f'Q{i+1}_avg_pnl': np.mean([t['pnl'] for t in q]),
        for i, q in enumerate(quartiles)
    }
```

**Prerequis** : trades tries chronologiquement (ils le sont deja dans `oos_trades_json`).

**Effort** : 0.5 jour.

#### B5.4 Composite Anti-Overfit Score -- PROJET

**Description** : Combiner plusieurs signaux d'overfitting en un score unique [0-100] :
- IS/OOS Calmar ratio (plus c'est proche de 1.0, mieux c'est)
- IS/OOS Win Rate delta
- IS/OOS PF ratio
- Perturbation degradation
- WFE (penalite si > 1.3 ou < 0.4)
- Nombre de trades OOS (plus c'est haut, plus c'est fiable)

Ce score serait le pendant "anti-overfit" du Quality Score existant.

**Effort** : 1 jour.

---

## Section C -- Roadmap suggeree

---

### v5.6 -- Quick Wins & Data Enrichment (2-3 semaines)

| # | Tache | Section | Effort | Impact |
|---|-------|---------|--------|--------|
| 1 | Omega Ratio + Tail Ratio | A1.1, A1.2 | 0.75 j | Haut |
| 2 | OOS Gate multi-niveaux (GOLD/SILVER/BRONZE) | A3.1 | 0.5 j | Haut |
| 3 | Exploit. avg_win/loss ratio + expectancy dans gate | B1.2, B1.3 | 0.5 j | Moyen |
| 4 | Analyse PnL distribution par config | B2.1 | 0.5 j | Moyen |
| 5 | Autocorrelation des trades | B2.2 | 0.25 j | Moyen |
| 6 | Metriques par exit reason | B2.4 | 0.25 j | Moyen |
| 7 | Run-length analysis | B2.5 | 0.25 j | Moyen |
| 8 | Score FTMO-readiness | B5.2 | 0.5 j | Haut |
| 9 | IS/OOS degradation vector | B5.1 | 0.5 j | Moyen |
| 10 | Long/Short ratio warning | B1.5 | 0.25 j | Faible |
| 11 | Stability decay index (quartile analysis) | B5.3 | 0.5 j | Moyen |
| 12 | Timestamps dans oos_trades_json | A5.2 | 0.5 j | Haut (prerequis) |
| 13 | Regime de marche dans les trades | A5.1 | 1.5 j | Haut |

**Total estime** : ~7 jours de dev
**Resultat** : Meilleur tri des configs, metriques plus riches, score FTMO, base enrichie pour v6.0.

---

### v5.7 -- WF & Perturbation Revolution (2-3 semaines)

| # | Tache | Section | Effort | Impact |
|---|-------|---------|--------|--------|
| 1 | Anchored Walk-Forward | A2.1 | 2 j | Haut |
| 2 | Perturbation graduee (5/10/15/20%) | A4.1 | 1.5 j | Haut |
| 3 | Perturbation des donnees | A4.2 | 2 j | Haut |
| 4 | Perturbation des couts de transaction | A4.3 | 0.5 j | Moyen |
| 5 | WF metriques supplementaires (Sharpe, DD) | A2.3 | 1 j | Moyen |
| 6 | Coherence IS/OOS dans le gate | A3.2 | 0.25 j | Moyen |
| 7 | Parameter convergence cross-runs | B3.1 | 1 j | Moyen |
| 8 | Composite anti-overfit score | B5.4 | 1 j | Moyen |
| 9 | Equity curve OOS stockee en DB | A5.5 | 1 j | Moyen |
| 10 | Regime detection (breakpoints) | B4.2 | 1.5 j | Moyen |

**Total estime** : ~12 jours de dev
**Resultat** : Walk-Forward professionnel, robustesse validee a 4 niveaux, detection de regime.

---

### v6.0 -- Performance & Multi-Instrument (4-6 semaines)

| # | Tache | Section | Effort | Impact |
|---|-------|---------|--------|--------|
| 1 | Vectorisation backtest + numba | A6.1 | 7 j | Tres haut |
| 2 | Cache indicateurs entre trials | A6.2 | 1 j | Moyen |
| 3 | Cross-instrument portfolio from DB | A7.1 | 3 j | Haut |
| 4 | WF avec re-optimisation par fenetre | A2.2 | 5 j | Tres haut |
| 5 | Meta-learning OOS prediction | B3.2 | 2 j | Haut |
| 6 | Config clustering | B3.3 | 1.5 j | Moyen |
| 7 | Universe screening auto | A7.2 | 2 j | Moyen |
| 8 | Template configs cross-instrument | A7.3 | 1 j | Moyen |
| 9 | PF par regime de marche | A1.3 | 2 j | Moyen |
| 10 | IS trades storage | A5.3 | 0.5 j | Moyen |
| 11 | Colonnes DB etendues (config_json -> SQL) | B1.4 | 2 j | Moyen |

**Total estime** : ~27 jours de dev
**Resultat** : Performance x5-10, vrai Walk-Forward avec re-optimisation, portfolio natif, ML predictif.

---

## Recapitulatif des priorites

### Top 10 des actions a plus haut impact/effort

| Rang | Action | Impact | Effort | ROI |
|------|--------|--------|--------|-----|
| 1 | OOS Gate multi-niveaux | Haut | 0.5 j | Tres haut |
| 2 | Score FTMO-readiness | Haut | 0.5 j | Tres haut |
| 3 | Timestamps dans trades | Haut (prerequis) | 0.5 j | Tres haut |
| 4 | Omega + Tail Ratio | Haut | 0.75 j | Haut |
| 5 | PnL distribution analysis | Moyen | 0.5 j | Haut |
| 6 | Anchored Walk-Forward | Haut | 2 j | Haut |
| 7 | Perturbation graduee | Haut | 1.5 j | Haut |
| 8 | Regime de marche dans trades | Haut | 1.5 j | Haut |
| 9 | Cross-instrument portfolio from DB | Haut | 3 j | Moyen-Haut |
| 10 | Vectorisation numba | Tres haut | 7 j | Moyen |

---

*Document genere par analyse complete de la documentation PF AI Lab 5.5 (5 fichiers MD) et de la base de donnees (40 runs, 800 configs, 49 023 trades OOS sur 9 instruments).*
