
Modelo base
Ramas de investigación a elegir
1. Aplicación de Explainable Boosting Models (EBM)
2. Uso de Machine Learning como filtro de señales en estrategias de Breakout
3. Stacking de modelos
4. Metalabeling (de Marcos Lopez de Prado)
5. Etiquetado Triple barrier
6. Modelos detectores de regímenes de mercado
Ayuda estado del arte. Modulo 2:
Ayuda data collection e instalacion paquetes environment:
Datos activos y etf:
1. Librería yfinance
Secciones  de interés para el Módulo 3:
1. Evaluación de modelos
2. Evaluación de las señales de trading
3. Interpretación del modelo e importancia características
4. Generación de señales out of sample
5. Backtesting vectorizado
Todos los notebooks adaptados a datos semanales
Referencia de TFM previo


Machine Learning aplicado a la gestión de Inversiones

Línea 1: Batir al índice de renta variable S&P 500 con la mejor selección de sectores. 
El trabajo está formado por los pasos necesarios para construir una estrategia de inversión rotacional semanal  donde habrá que predecir qué sectores lo harán mejor que el S&P 500. 

Modelo base


1.	Definición del problema y target de predicción para clasificación o regresión.


a.	En Regresión, el modelo predecirá el orden de los retornos en una regresión transversal para los etfs y la cartera entrada comprado del Top 3 y vendido del botom 3
b.	 Si se realiza clasificación habrá que definir un etiquetado cuando los retornos de los activos sean significativos o sean significativamente superiores al benchmark. Utilizaremos para la decisión de inversión en este caso la probabilidad que nos de el modelo de acierto.



2.	Obtención de datos con librerías públicas: yfinance, FRED api, pandas-datareader, edgar…etc

Tomaremos los datos de los ETFs sectoriales generales: Energía (XLE), Materiales (XLB) , Industria (XLI) , Tecnología (XLK), Financiero (XLF), Consumo Básico(XLP), Consumo Discrecional (XLY), Cuidados de la Salud (XLV), Servicios Publicos (XLU), Bienes Raíces (IYR), Servicios de Comunicación (VOX) y S&P (SPY)

XLE,XLB,XLI,XLK, XLF,XLP,XLY, XLV, XLU,  IYR, VOX ,SPY

Almacenamiento de datos en python y  uso en el modelo predictivo para una correcta generalización.


3.	Ingeniería de características predictivas:

Definición de variables con posible poder predictivo:
i.	Variables explicativas del ciclo económico: Inflación, PIB, Curva de tipos de interés
ii.	Factores comunes Fama-French: valor, crecimiento, momento, tamaño.
iii.	Otros factores: Indicadores de análisis técnico.
iv.	Análisis exploratorio de datos. Análisis de regímenes de mercado o ciclos económicos y funcionamiento de las características en cada ciclo (entrenamiento y fuera de muestra. Posiblemente necesarios 20 años de histórico)
Para esto os incluyo un par de enlaces muy utilies para enter la descripción de los ciclos y qué sectores lo han hecho mejor en estos ciclos.
https://www.fidelity.com/earning-center/trading-investing/markets-sectors/intro-sector-rotatioln-strats

Definición de los ciclos y fechas de los mismos
https://www.nber.org/research/business-cycle-dating

Deberéis observar el rendimiento de cada etf en cada fase del ciclo.
Tambien es importante que tengais unas nociones básicas de los factores que influyen
a cada etfs. Aquí teneis una referencia de las muchas que podeis encontrar por
internet  https://blog.capitaria.com/sectores-sp-etf

Entrenamiento en ciclo completo (o lo más completo posible)

Para esto vamos a reservar los últimos 4 años para periodo de fuera de muestra (desde el 2020 al 2024), los años desde 2008-2020 será el periodo de entrenamiento-validacion de hiperparametros, por lo que deberemos empezar con datos desde 2004 si es posible.

Luego nos iremos al notebook “model interpretation” para ver qué características resultan más predictivas en el periodo de validación. 





4.	Evaluación y selección de modelos de machine learning para entrenamiento.
a.	Random forest o gradient boosting
Notas: 
1.	Modelo para todos los etfs:
Primero vamos a intentar crear un modelo para todos los etfs, que sea generalizable y se apoye en regresión transversal (cross sectional). Entonces vamos a entrenar un modelo con características predictivas para todos los etfs. 


2.	Cross Validation de los modelos y Backtesting
Van a ir muy ligados estos dos pasos. Haremos cross validation para trabajar y observar las características  predictivas y si después de trabajarlo encontramos algo decente, lo podemos probar en un backtesting vectorizado 

Pasos del punto 2:
•	El primer paso walk-forward validation. https://i.sstatic.net/padg4.gif
•	Pararíamos y volveríamos a empezar si los resultados son malos
•	Si supera el primero, miramos si en out of sample también lo hace bien. 
o	Si lo hace muy mal, lo descartamos y volvemos a empezar (desde la ingeniería de características y su entendimiento, o hiperparametros o el target haya que revisarlo) Si lo hace decente, seguimos con el siguiente paso.
•	Miramos el backtesting vectorizado.
o	Si lo hace mal, revisaremos y volveremos a empezar, analizando las características predictivas.




6. Modelos detectores de regímenes de mercado
Identificar el estado actual del mercado (ej. alta volatilidad, tendencia alcista, etc.) permite adaptar las estrategias de inversión.
•	Paper clásico (Hidden Markov Models): "A Hidden Markov Model for detecting regime switches in financial time series"
o	Descripción: Los Modelos Ocultos de Márkov (HMM) son el enfoque clásico para este problema. Este paper es una buena introducción académica a su aplicación en finanzas.
o	Enlace: stacyzheng.github.io/files/hmm.pdf
•	Tutorial Práctico (Python): "Market Regime Detection Using Hidden Markov Models in Python"
o	Descripción: Un tutorial práctico que guía al lector a través de la implementación de un detector de regímenes con HMM usando librerías de Python.
o	Enlace: medium.com/@dcariboni/market-regime-detection-using-hidden-markov-models-in-python-5655511c2b3e
•	Enfoque moderno (Machine Learning): "Unsupervised Learning for Regime Detection"
o	Descripción: Artículo que explora el uso de algoritmos de clustering no supervisado (como los Gaussian Mixture Models) para identificar regímenes, una alternativa a los HMM.



Referencias importantes código:

https://github.com/stefan-jansen/machine-learning-for-trading

Libro fundamental 

Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and alternative data for systematic trading strategies with Python, 2nd Edition

https://amzn.eu/d/8xd3xrJ




Ayuda data collection e instalacion paquetes environment:

instalación paquetes en nuevo environment


•	conda install -c conda-forge zipline-reloaded
•	conda install -c conda-forge pyfolio-reloaded
•	conda install -c conda-forge alphalens-reloaded
•	conda install -c conda-forge empyrical-reloaded

Datos activos y etf:


1.	Librería yfinance

Vamos a tomar esta fuente de obtención de datos Open-High-Low-Close, por su simplicidad como la primera manera. 


Tratamiento, características, Factores Fama-French y almacenado

Aquí entre otras características encontraréis bajo el título de Rolling Factor Betas, las betas de Fama-French (con las betas nos será suficiente, no necesitamos calcular el risk premia que proponen otras secciones)

https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/04_alpha_factor_research/01_feature_engineering.ipynb

Utilizaremos la parte del código que utiliza “prices” que tendremos que cargar con los datos que previamente hayamos obtenido del punto 1.1. Hay otros datos de Market Cap, por ejemplo que no utilizaremos

Algunos datos Fred (características macroeconómicas)

https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/b662d5f933b48f2f02af62a23365e835e6334436/10_bayesian_machine_learning/02_pymc3_workflow.ipynb

 dato de ”unemployment rate“ https://fred.stlouisfed.org/series/UNRATE


Secciones  de interés para el Módulo 3:

A continuación os enumero notebooks del Github del libro que os proporciona código para elaborar todo el proceso de evaluación de características y generación de señales que podéis utilizar/ adaptar para el trabajo.
En el libro trabajan con datos diarios, pero yo os recomiendo que hagáis un resamaple para trabajar con semanales y así podáis reducir los tiempos de ejecución. Más abajo os doy un enlace donde os comparto estos notebooks adaptados para datos mensuales.



1.	Evaluación de modelos 

Ajustamos los hiper parámetros para el modelo (esta en código para LightGBM pero se puede adaptar también para random forest), utilizando 10 años para entrenamiento-validación y al menos 4 para test fuera de muestra.

Este es el notebook principal del que tenéis que utilizar el modelo de walkforward para hacer la validación. Esta es la parte principal que tenéis que ser capaces de integrar en vuestro código
.

https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/12_gradient_boosting_machines/05_trading_signals_with_lightgbm_and_catboost.ipynb
2.	Evaluación de las señales de trading

https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/12_gradient_boosting_machines/06_evaluate_trading_signals.ipynb


3.	Interpretación del modelo e importancia características 

https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/12_gradient_boosting_machines/07_model_interpretation.ipynb


4.	Generación de señales out of sample

https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/12_gradient_boosting_machines/08_making_out_of_sample_predictions.ipynb



5.	Backtesting vectorizado

https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/08_ml4t_workflow/02_vectorized_backtest.ipynb


Todos los notebooks adaptados a datos semanales

A continuación podéis encontrar todos los notebooks adaptados a datos mensuales donde se trabajan tanto random forest como lightgbm.  

He adaptado los del libro a nuestro ejercicio. Están revisados pero como siempre puede haber algún typo así que os recomiendo que los entendáis bien.

 2025

Referencia de TFM previo 

A continuación os dejo como referencia, para que podáis consultar la estructura de cada apartado, un trabajo previo.
https://openaccess.uoc.edu/items/1f208420-958a-424c-b3c3-f0af823e5ad5


