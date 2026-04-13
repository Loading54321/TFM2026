# Sector Rotation Strategy — TFM Código v2

Estrategia cuantitativa de rotación sectorial con Machine Learning y detección de
regímenes de mercado (Hidden Markov Model). Pipeline mensual, walk-forward estricto,
diseñado sin data leakage.

> **Frecuencia mensual**: el base_TFM recomienda frecuencia semanal, pero se trabaja
> a nivel mensual para reducir tiempos de ejecución y eliminar ruido de corto plazo,
> siguiendo la nota explícita del enunciado: *"os recomiendo que hagáis un resample
> para reducir los tiempos de ejecución"*. La elección mensual es estándar en la
> literatura de rotación sectorial (Moskowitz & Grinblatt 1999, Fama & French 1997).

---

## Estructura del proyecto

```
TFM codigo v2/
│
├── 01_data_download.py          Descarga ETFs (yfinance), macro (FRED), FF5
├── 02_feature_engineering.py    Construye panel long (date × etf) con features + target
├── 03_market_regime_detection.py Visualización HMM + CSV de respaldo + diagnósticos BIC
├── 04_walk_forward_training.py   Walk-forward LightGBM/RF/GB + HMM + EDA por régimen
├── 05_strategy_backtest.py       Backtest Long-Short Kelly, métricas vs SPY
├── 06_signal_evaluation.py       IC rolling, análisis quintiles, hit rate (Jansen cap.12)
│
├── config.py                    Todos los parámetros centralizados
├── models.py                    RandomForest, GradientBoosting; LightGBM configurado en 04
├── regime_model.py              GaussianHMM: fit, forward filter, Viterbi, diagnósticos
├── utils.py                     Carga de datos, get_feature_cols, helpers
│
├── run_all.py                   Ejecutar pipeline completo (02→04→03→05→06)
├── run_models_only.py           Ejecutar solo modelos (04→05, datos ya listos)
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
    ├── predictions_GradientBoosting.csv
    ├── eda_etf_by_regime.csv
    ├── eda_etf_by_regime.png
    ├── feature_importance_*.csv
    ├── signal_evaluation_IC_*.csv
    ├── signal_evaluation_quintiles_*.csv
    ├── signal_evaluation_plot.png
    └── backtest_chart.png
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
Frecuencia: **mensual**, 2000–2024

### Features (por ETF × mes)

| Grupo | Features |
|---|---|
| Técnicas ETF | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`, `momentum_12_1`, `vol_6m`, `spy_return` |
| Fama-French 5 | `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF` |
| Macro — Ciclo económico | `CPI_YoY`, `IndProd_YoY`, `GDP_YoY`, `Unemployment`, `Unemp_Chg` |
| Macro — Política monetaria | `FedFunds`, `FedFunds_Chg` |
| Tipos de interés | `T3M`, `T10`, `YieldSpread` (10yr-2yr), `Term_Spread_10_3m` (10yr-3m), `T10_Chg` |
| Riesgo mercado | `VIX`, `VIX_Chg`, `HY_OAS`, `HY_OAS_Chg` |
| Oro | `Gold_ret_1m`, `Gold_ret_3m` |
| HMM | `market_regime` (0=Bear, 1=Ranging, 2=Bull) |

**Target**: retorno real del ETF en el mes t+1 (shift −1 por ETF, sin leakage).

#### Nota sobre GDP_YoY (PIB real)
Serie FRED `GDPC1` (Real GDP, quarterly). Se descarga trimestral y se interpola
a mensual con **forward-fill** (causal: en cada mes sólo se usa el último dato
trimestral publicado). La variación interanual `GDP_YoY = pct_change(12)` captura
el ciclo económico tal como lo haría el indicador adelantado del ciclo de Fidelity
referenciado en el enunciado.

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
| Diagnósticos | — | BIC, log-likelihood, convergencia ✅ nuevo |

**Features HMM**: `ret_3m` (tendencia) + `vol_3m` (riesgo) del SPY.
La elección de dos features aproximadamente ortogonales (correlación negativa
por el *leverage effect*) evita el sesgo del EM que aparece con features
altamente correladas (problema documentado en la implementación con `ret_1m` + `ret_3m`).

- **Entrenamiento**: EM sobre IS completo (Jan 2008 – Dic 2019), parámetros fijos
- **Inferencia OOS**: filtro forward causal con ventana de contexto de **54 meses**
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

### Walk-Forward ML

- **Modelos**: LightGBM (principal), RandomForest y GradientBoosting (scikit-learn)
  - LightGBM: referenciado explícitamente en Jansen (2020), cap. 12, notebook 05;
    histogram-based, regularización L1/L2, más rápido que sklearn GB
- **Ventana ML**: expansiva desde `TRAIN_START` (Jan 2008) hasta t−1
- **OOS**: Jan 2020 – Dic 2024 (60 meses)
- **Modelo clonado** en cada iteración (sin estado compartido)
- **Cross-sectional**: un solo modelo entrenado con todos los ETFs a la vez

### Evaluación de señales — IC Analysis (Módulo 3, sección 2)

Script `06_signal_evaluation.py` basado en Jansen (2020), cap. 12, notebook 06:

- **IC mensual** (Information Coefficient = Spearman rank-correlation predicción/retorno real)
- **Rolling IC** (12 meses) — estabilidad temporal de la señal
- **ICIR** = mean(IC) / std(IC) — consistencia del IC
- **Análisis quintiles** — retorno medio real por grupo de predicción (Q1 → Q5)
- **Hit Rate** — % de veces que el signo de la predicción acierta
- **IC por régimen** — si el modelo funciona diferente en Bull/Ranging/Bear

Umbrales de referencia (Grinold & Kahn 2000, *Active Portfolio Management*):
- IC > 0.05 → señal débilmente útil; IC > 0.10 → moderadamente útil
- ICIR > 0.50 → IC consistente; Hit Rate > 55% → buena direccionalidad

### Estrategia de cartera — Long/Short con Kelly multi-activo

- **Long**: Top 3 ETFs por retorno predicho, ponderación Kelly multi-activo
- **Short**: Bottom 3 ETFs, ponderación Kelly multi-activo
- **Kelly**: `f* = Sigma_eff⁻¹ · m_eff` con half-Kelly (fracción 0.5)
- **Sigma**: covarianza histórica causal de 36 meses (sólo datos ≤ t)
- **Retorno mensual** = avg(long_t+1) − avg(short_t+1)  (dollar-neutral)
- **Rebalanceo mensual**, costos de transacción: 10 bps por operación

### Métricas de evaluación

CAGR, Volatilidad anualizada, Sharpe Ratio, Max Drawdown, Calmar Ratio  
Evaluación **IS** (2008–2019) y **OOS** (2020–2024) vs SPY benchmark

---

## Garantías anti-leakage

| Componente | Garantía |
|---|---|
| Features técnicas | Solo retornos y vol históricos hasta t |
| Macro FRED | Transformaciones YoY/diff sobre datos hasta t |
| GDP (GDPC1) | Forward-fill trimestral → mensual (causal) |
| HMM parámetros | EM entrenado solo en IS (<2020) |
| HMM regime_t | Forward filter con obs ≤ t (sin backward pass en 04) |
| ML training data | Siempre `date < t` (nunca incluye t ni futuro) |
| Target | `return.shift(-1)` por ETF — el retorno de t+1 nunca es feature |
| Backtest | Retorno de t+1: `prices.pct_change().shift(-1).loc[t]` |
| Kelly Sigma | `prices.loc[index <= t].tail(36)` — sólo datos causales |

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
python run_all.py   # 02 → 04 → 03 → 05 → 06
```

### 5. Re-ejecutar solo modelos (datos ya descargados)

```bash
python run_models_only.py   # 04 → 05
python 06_signal_evaluation.py   # evaluación de señales independiente
```

---

## Parámetros clave (`config.py`)

| Parámetro | Valor | Descripción |
|---|---|---|
| `TRAIN_START` | 2008-01-01 | Inicio entrenamiento ML |
| `TRAIN_END` | 2019-12-31 | Fin entrenamiento IS |
| `OOS_START` | 2020-01-01 | Inicio periodo out-of-sample |
| `OOS_END` | 2024-12-31 | Fin periodo out-of-sample |
| `DATA_START` | 2000-01-01 | Inicio descarga datos raw (>20 años) |
| `HMM_CONTEXT_MONTHS` | 54 | Ventana contexto HMM (meses) |
| `TOP_N` / `BOTTOM_N` | 3 / 3 | ETFs long / short por mes |
| `RANDOM_SEED` | 42 | Reproducibilidad |
| `N_ITER` (HMM) | 500 | Iteraciones EM (convergencia típica <200) |
| LightGBM `n_estimators` | 500 | Árboles LightGBM |
| LightGBM `learning_rate` | 0.05 | Igual que sklearn GB (comparación justa) |
| LightGBM `num_leaves` | 31 | Complejidad del árbol (~max_depth=5 sklearn) |

---

## Referencias

- Jansen, S. (2020). *Machine Learning for Algorithmic Trading* (2nd ed.). Packt.
  — github.com/stefan-jansen/machine-learning-for-trading
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*. McGraw-Hill.
- Fama, E. & French, K. (1997). Industry costs of equity. *Journal of Financial Economics*.
- Estrella, A. & Mishkin, F. (1998). Predicting U.S. recessions. *Review of Economics and Statistics*.
- Ryden, T. et al. (1998). A Hidden Markov Model for detecting regime switches.
  — stacyzheng.github.io/files/hmm.pdf
- QuantStart: Market Regime Detection using HMM in QSTrader.
  — quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader
- NBER Business Cycle Dating Committee — nber.org/research/business-cycle-dating
- Fidelity Sector Rotation — fidelity.com/earning-center/trading-investing/markets-sectors
