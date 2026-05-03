# Sector Rotation Strategy — TFM Código v2

> **Versión final de presentación — Mayo 2026**

Estrategia cuantitativa de rotación sectorial con Machine Learning y detección de
regímenes de mercado (Hidden Markov Model). Pipeline semanal, walk-forward estricto,
diseñado sin data leakage.

> **Frecuencia 100 % semanal**: los datos se procesan a frecuencia W-FRI (último
> viernes de cada semana), las predicciones del modelo son semanales y el portafolio
> se rebalancea **semanalmente** (cada viernes W-FRI).  Coherencia total entre el
> horizonte que aprende el modelo (retorno del ETF a 1 semana vs SPY) y el horizonte
> al que se invierte.  Métricas anualizadas con factor 52 (Sharpe × √52, CAGR con
> raíz 52-ésima).

---

## Estructura del proyecto

```
TFM codigo v2/
│
├── 01_data_download.py          Descarga ETFs (yfinance), macro (FRED + Gold + Oil), FF5
├── 02_feature_engineering.py    Construye panel long (date × etf) con features + target semanal
├── 03_market_regime_detection.py Visualización HMM + CSV de respaldo + diagnósticos BIC
├── 04_walk_forward_training.py   Walk-forward LightGBM/RF + HMM + EDA por régimen
├── 04b_regime_walk_forward.py    Walk-forward RegimeLGBM: LGBM con régimen HMM como feature
├── 05_strategy_backtest.py       Backtest Long-Short Kelly diagonal, métricas vs SPY
├── 06_signal_evaluation.py       IC rolling, análisis quintiles, hit rate (Jansen cap.12)
├── compare_strategies.py         Comparación anual OOS: LightGBM / RF / RegimeLGBM / SPY
│
├── config.py                    Todos los parámetros centralizados
├── models.py                    RandomForest y LightGBM (configurados en config.py)
├── regime_model.py              GaussianHMM: fit, forward filter, Viterbi, diagnósticos
├── utils.py                     Carga de datos, get_feature_cols, helpers
│
├── run_all.py                   Ejecutar pipeline completo (02→03→04→04b→05→06→C)
├── run_models_only.py           Ejecutar solo modelos (02→03→04→04b→05→06→C, sin descarga)
│
├── environment.yml              Dependencias conda
└── data/                        CSVs generados (no en control de versiones)
    ├── etf_prices.csv
    ├── fred_macro.csv
    ├── ff5_factors.csv
    ├── features_panel.csv
    ├── features_panel_with_regime.csv
    ├── market_regimes.csv
    ├── market_regimes_plot.png
    ├── predictions_LightGBM.csv
    ├── predictions_RandomForest.csv
    ├── predictions_RegimeLGBM.csv
    ├── eda_etf_by_regime.csv
    ├── eda_etf_by_regime.png
    ├── feature_importance_LightGBM.csv
    ├── feature_importance_RandomForest.csv
    ├── feature_importance_RegimeLGBM.csv
    ├── signal_evaluation_IC_*.csv
    ├── signal_evaluation_quintiles_*.csv
    ├── signal_evaluation_plot.png
    ├── backtest_chart.png
    └── comparison_chart.png
```

---

## Metodología

### Universo de activos

11 ETFs sectoriales S&P 500 (universo completo especificado en base_TFM):

| Ticker | Sector | Descripción |
|---|---|---|
| XLB | Materials | Materials Select Sector SPDR |
| XLE | Energy | Energy Select Sector SPDR |
| XLF | Financials | Financial Select Sector SPDR |
| XLI | Industrials | Industrial Select Sector SPDR |
| XLK | Technology | Technology Select Sector SPDR |
| XLP | Consumer Staples | Consumer Staples Select Sector SPDR |
| XLU | Utilities | Utilities Select Sector SPDR |
| XLV | Health Care | Health Care Select Sector SPDR |
| XLY | Consumer Disc. | Consumer Discretionary Select Sector SPDR |
| IYR | Real Estate | iShares U.S. Real Estate ETF |
| VOX | Comm. Services | Vanguard Communication Services ETF |

Benchmark: **SPY** (S&P 500)
Frecuencia: **semanal** (W-FRI), 2000–2024

### Features (por ETF × semana)

| Grupo | Features |
|---|---|
| Retornos ETF | `ret_1w`, `ret_3w`, `ret_4w`, `ret_7w`, `ret_8w`, `ret_13w`, `ret_26w`, `ret_52w` |
| Momentum | `momentum_52_4` (52w excluyendo últimas 4w) |
| Volatilidad | `vol_3w`, `vol_7w`, `vol_13w`, `vol_26w`, `vol_52w` (rolling anualizada) |
| RSI Wilder | `rsi_9w`, `rsi_14w`, `rsi_26w` (suavizado EWM causal) |
| Exceso vs SPY | `excess_ret_1w`, `excess_ret_13w`, `excess_ret_52w` |
| Rank cross-seccional | `ret_Xw_rank`, `vol_Xw_rank`, `rsi_Xw_rank` (percentil dentro del grupo esa semana) |
| SPY | `spy_ret_1w`, `spy_ret_4w`, `spy_ret_52w` |
| Fama-French 5 | `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF` |
| Macro — Ciclo económico | `CPI_YoY`, `IndProd_YoY`, `Unemployment`, `Unemp_Chg` |
| Macro — Política monetaria | `FedFunds`, `FedFunds_Chg` |
| Tipos de interés | `T3M`, `T10`, `YieldSpread` (10yr-2yr), `Term_Spread_10_3m`, `T10_Chg` |
| Crédito y riesgo | `VIX`, `VIX_Chg`, `HY_OAS`, `HY_OAS_Chg`, `IG_OAS`, `IG_OAS_Chg` |
| Bonos internacionales | `JGB10Y`, `JGB10Y_Chg`, `US_JP_Spread`, `RepoRate`, `RepoRate_Chg` |
| Materias primas | `Gold_ret_1w`, `Gold_ret_4w`, `Oil_ret_1w`, `Oil_ret_4w` |
| ISM / actividad | `ISM`, `ISM_Chg` (Chicago Fed NAI, proxy PMI manufacturero) |
| HMM (solo 04b) | `market_regime`, `bear_prob`, `ranging_prob`, `bull_prob` |

**Target**: exceso de retorno del ETF vs SPY en la semana t+1
(`return_ETF(t+1) − return_SPY(t+1)`, shift −1 por ETF, sin leakage).

> **Por qué exceso vs SPY**: la estrategia es de rotación sectorial. El objetivo es
> predecir qué sectores lo harán mejor que el mercado, no la dirección del mercado.
> Con retorno absoluto, el modelo aprende principalmente la dirección del SPY
> (igual para todos los ETFs en una semana dada) y no la rotación sectorial real.

---

### Detección de regímenes — GaussianHMM

Validado contra la literatura referenciada en el enunciado:

| Aspecto | Literatura (base_TFM) | Esta implementación |
|---|---|---|
| Modelo | GaussianHMM (hmmlearn) | GaussianHMM ✅ |
| N estados | 2–3 | **3** (Bear/Ranging/Bull) ✅ |
| Covarianza | `full` | `full` ✅ |
| Features | Retornos SPY | `ret_3m` + `vol_3m` (ortogonales) ✅ superior |
| Entrenamiento | Solo IS | Solo IS (parámetros fijos en OOS) ✅ |
| Inferencia OOS | Viterbi global (lookahead) | **Forward filter causal** ✅ superior |
| n_iter | 500–1000 | 500 con tol=1e-6 ✅ |
| Diagnósticos | — | BIC, log-likelihood, convergencia ✅ |

**Features HMM**: `ret_3m` (tendencia) + `vol_3m` (riesgo) del SPY.
La elección de dos features aproximadamente ortogonales (correlación negativa
por el *leverage effect*) evita el sesgo del EM que aparece con features
altamente correladas.

- **Entrenamiento**: EM sobre IS completo (Jan 2008 – Dic 2019), parámetros fijos
- **Inferencia OOS**: filtro forward causal con ventana rodante de **260 semanas** (~5 años)
  — prior uniforme al inicio de cada ventana, sin memoria de periodos más antiguos
- **Diagnósticos**: BIC, log-likelihood, convergencia EM, distribución de estados
- **Visualización** (`03`): Viterbi global sobre IS+OOS para gráfico limpio

### EDA por régimen de mercado

Análisis exploratorio ejecutado antes del walk-forward (requiere
`features_panel_with_regime.csv` de `03_market_regime_detection.py`):

- **Tabla de retorno medio** del exceso vs SPY por ETF y régimen (Bear / Ranging / Bull)
- **Líderes y rezagados** por fase del ciclo — responde directamente al requisito del enunciado
- **Salidas**: `data/eda_etf_by_regime.csv` y `data/eda_etf_by_regime.png`
- Usa exclusivamente el periodo IS (2008–2019), sin contaminación OOS

### Walk-Forward ML — Modelos globales (`04`)

- **Modelos**: LightGBM (principal), RandomForest (scikit-learn)
  - LightGBM: referenciado explícitamente en Jansen (2020), cap. 12, notebook 05;
    histogram-based, regularización L1/L2, más rápido que sklearn GB
- **Ventana ML**: expansiva desde `TRAIN_START` (Jan 2008) hasta t−1
- **OOS**: Jan 2020 – Dic 2024 (~260 semanas)
- **Exclusión opcional**: `ML_TRAIN_EXCLUDE_PERIODS` (p.ej. 2020–2021) para evitar
  que el modelo aprenda dinámicas COVID atípicas
- **Cross-seccional**: un solo modelo entrenado con todos los ETFs a la vez

### Walk-Forward RegimeLGBM — LGBM con régimen como feature (`04b`)

Un **único LightGBM** entrenado sobre todos los datos IS con cuatro features
adicionales que codifican el régimen HMM de forma causal:

| Feature | Descripción |
|---|---|
| `market_regime` | Régimen más probable (0=Bear, 1=Ranging, 2=Bull; −1 fuera de ventana HMM) |
| `bear_prob` | P(Bear \| observaciones pasadas) — forward filter causal |
| `ranging_prob` | P(Ranging \| observaciones pasadas) — forward filter causal |
| `bull_prob` | P(Bull \| observaciones pasadas) — forward filter causal |

**Por paso walk-forward (semana OOS t)**:
1. Ajustar GaussianHMM en ventana rodante [t − 260w, t) (EM/Baum-Welch)
2. Forward filter causal → probabilidades de régimen para cada semana de train
3. Añadir las 4 features de régimen al panel IS (probs uniformes 1/3 para semanas fuera de ventana)
4. Entrenar 1 LGBM con 73 features base + 4 features de régimen = **77 features en total**
5. Avanzar forward filter hasta t → régimen_t + probs_t
6. Inyectar régimen_t en pred_data y predecir con el LGBM

**Salidas**: `predictions_RegimeLGBM.csv` (columnas: date, etf, predicted_return, rank,
target, regime, regime_name, bear_prob, ranging_prob, bull_prob)

### Evaluación de señales — IC Analysis (Módulo 3, sección 2)

Script `06_signal_evaluation.py` basado en Jansen (2020), cap. 12, notebook 06:

- **IC semanal** (Information Coefficient = Spearman rank-correlation predicción/retorno real)
- **Rolling IC** (52 semanas ≈ 1 año) — estabilidad temporal de la señal
- **ICIR** = mean(IC) / std(IC) — consistencia del IC
- **Análisis quintiles** — retorno medio real por grupo de predicción (Q1 → Q5)
- **Hit Rate** — % de veces que el signo de la predicción acierta
- **IC por régimen** — si el modelo funciona diferente en Bull/Ranging/Bear

Umbrales de referencia (Grinold & Kahn 2000, *Active Portfolio Management*):
- IC > 0.05 → señal débilmente útil; IC > 0.10 → moderadamente útil
- ICIR > 0.50 → IC consistente; Hit Rate > 55% → buena direccionalidad

### Estrategia de cartera — Long/Short con Kelly diagonal

- **Long**: Top-`TOP_N` ETFs por retorno predicho, ponderación Kelly diagonal por activo
- **Short**: Bottom-`BOTTOM_N` ETFs con **filtro doble** (pred < 0 AND |pred| > pred_Top1),
  peso = 0 si no cumple ambas condiciones (long-only parcial esa semana)
- **Kelly diagonal**: `k_i = KELLY_FRACTION × |pred_i| / var_i` (por activo, sin matriz de covarianza)
- **var_i**: varianza histórica causal de `KELLY_LOOKBACK_WEEKS` semanas (solo datos ≤ t)
- **Retorno semanal** = Σ(w_long × ret_long_t+1) − Σ(|w_short| × ret_short_t+1)  (dollar-neutral)
- **Rebalanceo semanal** (cada viernes W-FRI); costes de transacción: `COST_BPS` bps por
  nombre entrado/salido (normalizado por posiciones mantenidas)
- Aplicados también a los benchmarks EW para que la comparación entre estrategias sea justa

### Comparación de estrategias (`compare_strategies.py`)

Tabla y gráfico comparativo anual OOS:

| Estrategia | Descripción |
|---|---|
| LightGBM | Modelo global, Kelly diagonal, Long-Short |
| RandomForest | Modelo global, Kelly diagonal, Long-Short |
| RegimeLGBM | LGBM con régimen HMM como feature, Kelly diagonal, Long-Short |
| X Top3 | Modelo base, solo pata larga (Top-N, Kelly diagonal Long-Only) |
| Top-3 EW | Top-N por predicted_return, pesos iguales, con mismos costes `COST_BPS` |
| SPY | Benchmark pasivo Buy & Hold |

### Métricas de evaluación

CAGR, Volatilidad anualizada, Sharpe Ratio, Max Drawdown, Calmar Ratio
Evaluación **IS** (2008–2019) y **OOS** (2020–2024) vs SPY benchmark

---

## Garantías anti-leakage

| Componente | Garantía |
|---|---|
| Features técnicas | Solo retornos y vol históricos hasta t |
| Rank cross-seccional | Calculado con corte transversal de la semana t (no OOS) |
| Exceso vs SPY | `ret_etf(t) − ret_spy(t)`, ambos realizados en t |
| Macro FRED | Transformaciones YoY/diff sobre datos hasta t |
| HMM parámetros | EM entrenado solo con datos `< t` en cada paso walk-forward |
| HMM regime_t | Forward filter causal con observación de t (sin backward pass) |
| Features régimen train | Forward filter causal por semana; probs uniformes para datos fuera de ventana |
| ML training data | Siempre `date < t` (nunca incluye t ni futuro) |
| Target | `excess_return.shift(-1)` por ETF — el retorno de t+1 nunca es feature |
| Backtest | Retorno de t+1: `weekly_forward_returns(prices, dates).loc[t]` (shift(-1)) |
| Kelly var_i | `prices.loc[index <= t].tail(KELLY_LOOKBACK_WEEKS)` — solo datos causales |
| Turnover tx | Comparación con portfolio de t-1 (nunca con t+1) |

---

## Instalación y ejecución

### 1. Crear entorno

```bash
conda env create -f environment.yml
conda activate tfm-ml-trading
```

### 2. Configurar API key de FRED

En el archivo `.env` (ya incluido en el proyecto, **no subir a git**):
```
FRED_API_KEY=tu_api_key_aqui
```
Obtener gratis en: https://fred.stlouisfed.org/docs/api/api_key.html

### 3. Descargar datos (solo primera vez)

```bash
python 01_data_download.py
```

### 4. Ejecutar pipeline completo

```bash
python run_all.py
# Orden: 02 → 03 → 04 → 04b → 05 → 06 → compare_strategies
```

### 5. Re-ejecutar solo modelos (datos ya descargados)

```bash
python run_models_only.py
# Orden: 02 → 03 → 04 → 04b → 05 → 06 → compare_strategies
```

### 6. Ejecutar scripts individuales

```bash
python 04b_regime_walk_forward.py  # solo modelo RegimeLGBM
python 06_signal_evaluation.py     # solo evaluación de señales
python compare_strategies.py       # solo tabla comparativa
```

---

## Parámetros clave (`config.py`)

| Parámetro | Valor | Descripción |
|---|---|---|
| `DEV_MODE` | True/False | True → parámetros reducidos (~2-3 min), False → TFM final (~8-10 min) |
| `TRAIN_START` | 2008-01-01 | Inicio entrenamiento ML |
| `TRAIN_END` | 2019-12-31 | Fin entrenamiento IS |
| `OOS_START` | 2020-01-01 (2023 en dev) | Inicio periodo out-of-sample |
| `OOS_END` | 2024-12-31 | Fin periodo out-of-sample |
| `DATA_START` | 2000-01-01 | Inicio descarga datos raw (>20 años) |
| `HMM_REGIME_LOOKBACK` | 260 semanas (~5 años) | Ventana rodante HMM en 04b |
| `HMM_CONTEXT_PERIODS` | 235 semanas (~4.5 años) | Ventana forward filter HMM en 04 |
| `ML_TRAIN_EXCLUDE_PERIODS` | [(2020-01, 2021-12)] | Periodos excluidos del entrenamiento ML |
| `TOP_N` / `BOTTOM_N` | 3 / 3 | ETFs long / short por semana |
| `KELLY_FRACTION` | 1.0 | Kelly completo (fracción del Kelly) |
| `KELLY_LOOKBACK_WEEKS` | 36 | Ventana causal (semanas) para var_i en Kelly |
| `COST_BPS` | 10 | Basis points por leg en costes de transacción |
| `RANDOM_SEED` | 42 | Reproducibilidad |
| RF `n_estimators` | 50 (dev) / 200 (final) | Árboles Random Forest |
| LightGBM `n_estimators` | 100 (dev) / 500 (final) | Árboles LightGBM |
| LightGBM `learning_rate` | 0.05 | Tasa de aprendizaje |
| LightGBM `num_leaves` | 31 | Complejidad del árbol (~max_depth=5) |

---

## Referencias

- Jansen, S. (2020). *Machine Learning for Algorithmic Trading* (2nd ed.). Packt.
  — github.com/stefan-jansen/machine-learning-for-trading
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*. McGraw-Hill.
- Fama, E. & French, K. (1997). Industry costs of equity. *Journal of Financial Economics*.
- Estrella, A. & Mishkin, F. (1998). Predicting U.S. recessions. *Review of Economics and Statistics*.
- Thorp, E. (2006). *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*.
- Ryden, T. et al. (1998). A Hidden Markov Model for detecting regime switches.
  — stacyzheng.github.io/files/hmm.pdf
- QuantStart: Market Regime Detection using HMM in QSTrader.
  — quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader
- Moskowitz, T. & Grinblatt, M. (1999). Do industries explain momentum? *Journal of Finance*.
- NBER Business Cycle Dating Committee — nber.org/research/business-cycle-dating
- Fidelity Sector Rotation — fidelity.com/earning-center/trading-investing/markets-sectors
