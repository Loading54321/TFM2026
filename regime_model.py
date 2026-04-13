"""
regime_model.py
===============
Funciones compartidas para la deteccion de regimenes con Gaussian HMM.

Frecuencia: semanal (W-FRI). Las observaciones son ret_13w y vol_13w,
equivalente semanal a ret_3m/vol_3m (13 semanas ≈ 1 trimestre).

Importadas por:
  03_market_regime_detection.py  (visualizacion + CSV de respaldo)
  04_walk_forward_training.py    (regimen integrado en el walk-forward)

─────────────────────────────────────────────────────────────────────────────
VALIDACION CONTRA LITERATURA
─────────────────────────────────────────────────────────────────────────────

[1] Paper clasico (HMM):
    "A Hidden Markov Model for detecting regime switches in financial
    time series" — stacyzheng.github.io/files/hmm.pdf
    -> El paper usa GaussianHMM con retornos como observaciones y
       covariance_type='full'. Esta implementacion CUMPLE y MEJORA:
       usa (ret_13w, vol_13w) en lugar de solo retornos, separando mejor
       los estados de alta y baja volatilidad.

[2] Tutorial practico:
    "Market Regime Detection using Hidden Markov Models in Python"
    — Cariboni / QuantStart (GaussianHMM + hmmlearn)
    Codigo de referencia QuantStart:
      hmm_model = GaussianHMM(n_components=2, covariance_type='full',
                              n_iter=1000).fit(rets)
    Esta implementacion MEJORA en:
      a) 3 estados (Bear/Ranging/Bull) vs 2 del tutorial basico —
         mas robusto e interpretable (validado en literatura: PyQuantLab 2025)
      b) n_iter=500 con tol=1e-6 (convergencia mas rigurosa)
      c) Inicializacion economica explicita (evita minimos locales del EM)
      d) Forward filter causal para OOS vs Viterbi global con lookahead

[3] Enfoque moderno (ML no supervisado):
    "Unsupervised Learning for Regime Detection" — Gaussian Mixture Models
    -> GaussianHMM incluye la dinamica temporal de los GMM; es el enfoque
       canonico para series financieras con persistencia de estados.

─────────────────────────────────────────────────────────────────────────────
ELECCION DE FEATURES — por que (ret_13w, vol_13w):
─────────────────────────────────────────────────────────────────────────────

  Problema con (ret_1w, ret_13w):
    - ret_13w ~ suma de ret_1w[t]...ret_1w[t-12]  -> correlacion alta
    - Usar covariance_type='diag' con features correladas es matematicamente
      incorrecto: el modelo asume independencia donde no la hay, y la EM
      converge a minimos locales donde Bear captura demasiados periodos.

  Solucion — dos features aproximadamente ortogonales:
    ret_13w  retorno acumulado 13 semanas del SPY  (senial de DIRECCION)
    vol_13w  volatilidad realizada 13w, anualizada (senial de RIESGO)
    -> Alineado con PyQuantLab (2025): "returns + volatility" como
       observaciones standard para HMM de regimenes financieros.
    -> 13 semanas ≈ 3 meses: la escala de retorno compuesto es identica,
       por lo que las medias iniciales se mantienen iguales al caso mensual.

  Separacion en el espacio (ret_13w, vol_13w):
    Bear:    ret_13w << 0,  vol_13w >> alto  (caidas violentas + panico)
    Ranging: ret_13w ~ 0,  vol_13w medio    (lateralizacion)
    Bull:    ret_13w > 0,  vol_13w bajo     (tendencia sostenida + calma)

  La volatilidad es el discriminante mas potente: Bear tiene vol 3-4x
  mayor que Bull, mientras que en el espacio de retornos solos los tres
  estados se solapan significativamente.

  covariance_type='full': permite al EM estimar la covarianza negativa
  real entre retorno y volatilidad (efecto palanca / leverage effect).

─────────────────────────────────────────────────────────────────────────────
REGIMENES: 0=Bear | 1=Ranging | 2=Bull  (ordenados por ret_13w medio)

DECODIFICACION:
  IS  -> Viterbi global (secuencia optima, sin lookahead fuera del IS)
  OOS -> Forward filter causal, un paso por semana (sin ninguna info futura)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM
from config import DATA_DIR, RANDOM_SEED

N_STATES      = 3
N_ITER        = 500
REGIME_NAMES  = {0: "Bear", 1: "Ranging", 2: "Bull"}
REGIME_COLORS = {0: "#e63946", 1: "#adb5bd", 2: "#2dc653"}

# Nombres de las dos observaciones del HMM (usados en diagnosticos y seleccion)
OBS_COLS = ["ret_13w", "vol_13w"]


# ── 1. Features ───────────────────────────────────────────────────────────────

def load_spy_features(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Carga precios semanales del SPY (W-FRI) y calcula las dos observaciones del HMM:

      ret_13w  retorno acumulado 13 semanas  pct_change(13)
               -> capta la DIRECCION/tendencia trimestral; equivalente a ret_3m mensual

      vol_13w  volatilidad realizada 13 semanas, anualizada
               std(ret_1w ultimas 13 semanas) * sqrt(52)
               -> capta el RIESGO / nivel de panico del mercado

    Las dos features son aproximadamente ortogonales: la correlacion entre
    retorno y volatilidad existe (leverage effect) pero no es degenerada,
    por lo que el HMM puede separar bien los tres regimenes.

    Se eliminan las primeras filas con NaN (ventana de 13 semanas inicial).
    """
    prices = pd.read_csv(
        f"{data_dir}/etf_prices.csv", index_col=0, parse_dates=True
    )
    prices.index.name = "date"
    prices.sort_index(inplace=True)

    spy     = prices["SPY"]
    ret_1w  = spy.pct_change()
    ret_13w = spy.pct_change(13)
    vol_13w = ret_1w.rolling(13).std() * np.sqrt(52)   # volatilidad anualizada

    df = pd.DataFrame({"ret_13w": ret_13w, "vol_13w": vol_13w}, index=spy.index)
    df.dropna(inplace=True)
    return df


# ── 2. Ajuste del HMM ────────────────────────────────────────────────────────

def fit_hmm(X_train: np.ndarray) -> GaussianHMM:
    """
    Ajusta GaussianHMM(n=3, covariance='full') con inicializacion economica.

    Por que covariance_type='full':
      Con features correladas (incluso ret_3m y vol_3m tienen correlacion
      negativa por el leverage effect), asumir covarianza diagonal introduce
      sesgo en la estimacion de las densidades de emision.  'full' deja que
      el EM estime la covarianza real de cada estado.

    Por que init_params='':
      Desactiva la inicializacion aleatoria de hmmlearn y usa exactamente
      los valores economicos que definimos a continuacion.  Evita minimos
      locales donde todos los estados colapsan al mismo cluster.

    Inicializacion en el espacio (ret_13w, vol_13w):
      Valores en decimal; vol_13w es anualizada (ej. 0.40 = 40% vol anual).
      ret_13w ≈ ret_3m en escala de retorno compuesto (13 semanas ≈ 3 meses).

      Bear:    ret_13w ~ -18%,  vol_13w ~ 42%  (crisis: caida + panico)
      Ranging: ret_13w ~  +2%,  vol_13w ~ 18%  (mercado lateral)
      Bull:    ret_13w ~  +8%,  vol_13w ~ 12%  (tendencia alcista tranquila)
    """
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        init_params="",     # no inicializacion aleatoria
        tol=1e-6,
        verbose=False,
    )

    # Distribucion inicial: Bear raro, Bull mas comun historicamente
    model.startprob_ = np.array([0.15, 0.25, 0.60])

    # Matriz de transicion: alta persistencia (mercados son persistentes)
    model.transmat_ = np.array([
        [0.85, 0.10, 0.05],   # desde Bear   -> mayor prob de seguir Bear
        [0.10, 0.75, 0.15],   # desde Ranging -> puede pasar a cualquier estado
        [0.04, 0.10, 0.86],   # desde Bull   -> mayor prob de seguir Bull
    ])

    # Medias: [ret_13w, vol_13w]
    # ret_13w tiene la misma escala que ret_3m (13 semanas ≈ 3 meses en retorno compuesto)
    # vol_13w es anualizada: misma escala que vol_3m mensual anualizada
    model.means_ = np.array([
        [-0.18,  0.42],   # Bear
        [ 0.02,  0.18],   # Ranging
        [ 0.08,  0.12],   # Bull
    ])

    # Covarianzas completas 2x2, inicializadas como diagonales.
    # El EM aprendera los terminos fuera de la diagonal (covarianza negativa
    # esperada por el leverage effect: cuando ret cae, vol sube).
    model.covars_ = np.array([
        np.diag([0.12**2, 0.16**2]),   # Bear:    alta dispersion en ambas dims
        np.diag([0.07**2, 0.07**2]),   # Ranging: moderada
        np.diag([0.05**2, 0.04**2]),   # Bull:    baja dispersion
    ])

    model.fit(X_train)
    return model


def label_mapping(model: GaussianHMM) -> dict:
    """
    Reasigna indices HMM (arbitrarios tras EM) a etiquetas economicas
    ordenando por la media de ret_13w (primera feature):
      menor ret_13w -> Bear   (0)
      medio         -> Ranging (1)
      mayor         -> Bull   (2)
    Devuelve {indice_hmm: etiqueta_economica}.
    """
    sorted_states = np.argsort(model.means_[:, 0])   # ordena por ret_13w
    return {int(hmm_s): int(econ_l) for econ_l, hmm_s in enumerate(sorted_states)}


# ── 3. Probabilidad de emision (covarianza completa) ─────────────────────────

def _emission(model: GaussianHMM, obs: np.ndarray) -> np.ndarray:
    """
    P(obs | estado) para GaussianHMM con covariance_type='full'.

    Calcula la densidad gaussiana multivariada para cada estado:
      log p = -0.5 * ( (obs-mu)^T Sigma^{-1} (obs-mu) + log|Sigma| + k*log(2pi) )

    Se aniade una regularizacion minima (1e-8 * I) para garantizar que
    Sigma sea definida positiva incluso si el EM converge a matrices casi
    singulares (raro con 'full' y suficientes datos, pero defensivo).
    """
    k  = len(obs)
    em = np.zeros(model.n_components)
    for s in range(model.n_components):
        diff    = obs - model.means_[s]
        cov     = model.covars_[s] + np.eye(k) * 1e-8   # regularizacion
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:          # no deberia ocurrir con regularizacion
            em[s] = 1e-300
            continue
        cov_inv = np.linalg.inv(cov)
        log_p   = -0.5 * (diff @ cov_inv @ diff + logdet + k * np.log(2 * np.pi))
        # Solo prevenimos underflow (log_p < -500 → exp → 0 exacto en float64).
        # NO ponemos upper bound: una Gaussiana estrecha tiene pdf > 1 en el modo
        # (es densidad, no probabilidad), y capear en exp(0)=1 comprime el ratio
        # entre estados exactamente cuando la discriminación debería ser máxima.
        em[s]   = np.exp(np.clip(log_p, -500, 500))
    return np.maximum(em, 1e-300)


# ── 4. Algoritmo forward (causal) ────────────────────────────────────────────

def forward_step(
    model: GaussianHMM,
    alpha: np.ndarray,
    obs: np.ndarray,
) -> np.ndarray:
    """
    Un paso del filtro forward (puramente causal):
      alpha_t = normalize( (alpha_{t-1} @ A) * b(obs_t) )

    Donde:
      alpha_{t-1}  distribucion de estado en t-1  (vector de longitud N_STATES)
      A            matriz de transicion del HMM
      b(obs_t)     verosimilitud de la observacion en cada estado

    alpha_t depende unicamente de datos hasta t (sin backward pass ni lookahead).
    Se llama semana a semana dentro del loop de 04_walk_forward_training.py.
    """
    em        = _emission(model, obs)
    alpha_new = (alpha @ model.transmat_) * em
    total     = alpha_new.sum()
    if total > 1e-300:
        return alpha_new / total
    return np.ones(model.n_components) / model.n_components


def get_alpha_end_of_is(model: GaussianHMM, X_is: np.ndarray) -> np.ndarray:
    """
    Corre el filtro forward sobre toda la serie IS y devuelve alpha al final.
    Este vector es el punto de partida causal para el OOS en el walk-forward.
    """
    alpha = model.startprob_.copy()
    for obs in X_is:
        alpha = forward_step(model, alpha, obs)
    return alpha


# ── 5. Decodificacion ─────────────────────────────────────────────────────────

def decode_is_states(
    model: GaussianHMM,
    X_is: np.ndarray,
    mapping: dict,
) -> np.ndarray:
    """
    Viterbi sobre datos IS unicamente.

    hmmlearn.predict() implementa el algoritmo de Viterbi: encuentra la
    secuencia de estados S* = argmax P(S | X, model) de forma global.
    Es mas estable que argmax del forward filter porque considera toda la
    secuencia IS al mismo tiempo (sin lookahead fuera del IS).
    """
    raw = model.predict(X_is)
    return np.vectorize(mapping.get)(raw).astype(int)


def decode_full_viterbi(
    model: GaussianHMM,
    X_all: np.ndarray,
    mapping: dict,
) -> np.ndarray:
    """
    Viterbi sobre la secuencia COMPLETA (IS + OOS).

    Usado por 03_market_regime_detection.py unicamente para VISUALIZACION.
    Da la secuencia de regimenes mas suave e interpretable porque el algoritmo
    Viterbi considera toda la historia al decodificar (backward pass global).

    NO se usa en el walk-forward (04): ahi el filtro forward causal garantiza
    que cada regimen_t no depende de observaciones futuras a t.

    Nota sobre leakage:
      El modelo se entrena SOLO con X_is. El Viterbi se aplica a IS+OOS
      usando el modelo ya fijado.  La secuencia OOS se decodifica con
      informacion futura dentro de OOS (backward pass de Viterbi), pero
      el modelo en si no se reentrena con OOS. Para el TFM esto es aceptable
      en el contexto de visualizacion/diagnostico.
    """
    raw = model.predict(X_all)
    return np.vectorize(mapping.get)(raw).astype(int)


def decode_oos_causal(
    model: GaussianHMM,
    X_is: np.ndarray,
    X_oos: np.ndarray,
    mapping: dict,
) -> np.ndarray:
    """
    Forward filter causal para el OOS completo (alternativa al Viterbi OOS).
    Necesario en 04 para garantizar no lookahead en predicciones.
    Tambien exportado para compatibilidad con codigo existente.
    """
    alpha  = get_alpha_end_of_is(model, X_is)
    labels = np.zeros(len(X_oos), dtype=int)
    for i, obs in enumerate(X_oos):
        alpha     = forward_step(model, alpha, obs)
        labels[i] = mapping[int(np.argmax(alpha))]
    return labels


# ── 6. Diagnostico del modelo (BIC, log-likelihood, convergencia) ────────────

def hmm_diagnostics(model: GaussianHMM, X_train: np.ndarray) -> dict:
    """
    Metricas de calidad del HMM ajustado, alineadas con la literatura.

    Log-likelihood:
      Mide que tan bien el modelo explica los datos de entrenamiento IS.
      Valor esperado para datos financieros mensuales con ~144 obs: > -200.

    BIC (Bayesian Information Criterion):
      BIC = -2 * log-lik + k * log(n)
      donde k = numero de parametros libres del modelo.
      Para GaussianHMM(n=3, cov='full', obs_dim=2):
        k = (n-1)         # prob iniciales (startprob)
          + n*(n-1)       # transmat (n filas, cada una suma a 1 -> n-1 libres)
          + n*d           # medias (d=2 features por estado)
          + n*d*(d+1)/2   # covarianzas full simétricas
        = 2 + 6 + 6 + 9 = 23 parametros
      BIC mas bajo = mejor ajuste penalizado por complejidad.

    Convergencia:
      Si model.monitor_.converged es False, la EM no alcanzo tol=1e-6
      en N_ITER=500 iteraciones. En la practica, con inicializacion
      economica correcta, la EM suele converger antes de 200 iteraciones.

    Distribucion de regimenes IS:
      Proporcion esperada historicamente (1990-2020, S&P 500):
        Bull   ~60-65% de los meses
        Ranging ~20-25%
        Bear   ~12-18%
      Proporciones muy distintas indican colapso de estados (local minima).
    """
    log_lik  = model.score(X_train)
    n, d     = X_train.shape
    # Numero de parametros libres (GaussianHMM full, n_states=3, obs_dim=2)
    n_states = model.n_components
    k_params = (
        (n_states - 1)                       # startprob
        + n_states * (n_states - 1)          # transmat
        + n_states * d                       # means
        + n_states * d * (d + 1) // 2       # covars full
    )
    bic = -2 * log_lik + k_params * np.log(n)

    # Distribucion de estados via Viterbi IS
    raw_states = model.predict(X_train)
    unique, counts = np.unique(raw_states, return_counts=True)
    state_pct = {int(s): float(c / n) for s, c in zip(unique, counts)}

    return {
        "log_likelihood" : round(log_lik, 2),
        "bic"            : round(bic, 2),
        "n_params"       : k_params,
        "n_obs"          : n,
        "converged"      : model.monitor_.converged,
        "n_iter_done"    : model.monitor_.iter,
        "state_pct_raw"  : state_pct,   # indices HMM (antes del label_mapping)
    }


def get_regime_from_context_window(
    model: GaussianHMM,
    spy_df: pd.DataFrame,
    t: pd.Timestamp,
    window_periods: int,
    mapping: dict,
) -> int:
    """
    Determina el regimen para la semana t usando una ventana de contexto rodante.

    Logica:
      1. Toma las ultimas `window_periods` observaciones semanales de spy_df hasta t
         (inclusive), es decir solo datos disponibles causalmente en t.
      2. Inicia el filtro forward con prior uniforme (sin memoria de antes
         de la ventana), lo que hace la estimacion del regimen mas adaptable
         a cambios estructurales recientes.
      3. Devuelve argmax(alpha_t) tras recorrer toda la ventana.

    Por que prior uniforme en lugar de continuar el alpha anterior:
      Continuar el alpha desde el principio de la serie arrastra informacion
      de regimenes muy antiguos que pueden no ser relevantes para el mercado
      actual.  Con una ventana de 235 semanas (≈ 4.5 años), el filtro solo
      considera historia reciente, lo que equivale a 'preguntar' al HMM
      cual es el regimen mas probable dado el comportamiento reciente del SPY.

    Los PARAMETROS del HMM (means_, covars_, transmat_) siguen siendo los
    estimados en IS con EM.  Solo la inferencia del estado cambia: en lugar
    de una cadena causal desde t=0, se usa una ventana deslizante.

    Causalidad garantizada: spy_df.loc[spy_df.index <= t] usa solo datos
    disponibles en t; no hay observaciones futuras en X_window.
    """
    available = spy_df.loc[spy_df.index <= t, OBS_COLS]
    if available.empty:
        return 1   # fallback: Ranging si no hay datos suficientes

    X_window = available.iloc[-window_periods:].values   # ultimos window_periods

    # Prior uniforme: sin bias hacia ningun estado al inicio de la ventana
    alpha = np.ones(model.n_components) / model.n_components
    for obs in X_window:
        alpha = forward_step(model, alpha, obs)

    return mapping[int(np.argmax(alpha))]
