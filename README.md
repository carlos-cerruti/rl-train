# 🚀 LightGBM ML Scoring System for Trading Indices

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Un sistema avanzado de Machine Learning que evalúa la probabilidad de éxito de setups de trading en **índices (US30, US100, US500)** utilizando LightGBM con optimización GPU y Optuna para hiperparámetros.

## 🎯 Características Principales

- **Optimización Automática**: Descubre y optimiza automáticamente tanto los parámetros de indicadores técnicos como los hiperparámetros del modelo
- **Multi-Timeframe**: Analiza confluencia entre timeframes (D1, H4, H1, M15)
- **Indicadores Clave**: Fibonacci, MACD, RVI, RSI, ATR, Volumen y Divergencias
- **Scoring en Tiempo Real**: Clase `SetupScorer` para evaluación en vivo
- **GPU Optimizado**: Aprovecha LightGBM con aceleración GPU
- **Pipeline Completo**: Desde datos históricos hasta scoring en producción

## 🏗️ Arquitectura del Sistema

```
📊 Datos Históricos → 🔧 Feature Engineering → 🎯 Optimización Optuna → 📈 Modelo LightGBM → ⚡ Scoring Real-Time
```

### Componentes Principales

1. **Generador de Dataset Histórico** (`data_sources/`)
   - Descarga 3-5 años de datos para US30, US100, US500
   - Soporte para Yahoo Finance y MetaTrader 5
   - Múltiples timeframes simultáneos

2. **Feature Engineering Dinámico** (`features.py`)
   - Fibonacci: Proximidad a niveles de retroceso
   - MACD: Tendencia y momentum en H4
   - RVI: Timing de entrada en M15
   - RSI: Filtros de sobrecompra/venta en H1
   - Divergencias: Detección multi-timeframe
   - Contexto temporal y de sesión

3. **Optimización con Optuna** (`optimization/`)
   - Optimiza simultáneamente parámetros de features y modelo
   - Cross-validation temporal para series de tiempo
   - Pruning automático de trials poco prometedores
   - GPU acelerado con LightGBM

4. **Scoring en Tiempo Real** (`scoring/`)
   - Clase `SetupScorer` para evaluación instantánea
   - Análisis de confluencia multi-timeframe
   - Evaluación de riesgo y fuerza de señal

## 🚀 Instalación Rápida

```bash
# Clonar el repositorio
git clone <your-repo-url>
cd lightbgm

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

### Dependencias Principales

- **ML**: `lightgbm`, `optuna`, `scikit-learn`
- **Datos Financieros**: `yfinance`, `MetaTrader5`, `financedatabase`
- **Indicadores Técnicos**: `ta`, `talib`
- **Procesamiento**: `pandas`, `numpy`, `pyarrow`

## 📖 Uso Rápido

### 1. Pipeline Completo (Recomendado para empezar)

```python
from lightbgm.modeling.train import full_pipeline

# Ejecuta todo: datos → features → optimización → modelo final
full_pipeline(n_trials=100, start_year=2020, end_year=2024)
```

### 2. Paso a Paso

```python
# Generar datos históricos
from lightbgm.modeling.train import generate_data
generate_data(symbols="US30,US100,US500", start_year=2020)

# Optimizar modelo
from lightbgm.modeling.train import optimize
optimize(n_trials=500)  # Más trials = mejor optimización

# Usar el scorer entrenado
from lightbgm.scoring.setup_scorer import create_scorer_from_artifacts
scorer = create_scorer_from_artifacts("models/optimization_results")

# Scoring en tiempo real
result = scorer.score_setup(market_data, "US30", "H4")
print(f"Probabilidad de éxito: {result.score:.2%}")
print(f"Confianza: {result.confidence}")
```

### 3. Análisis Multi-Timeframe

```python
# Datos para diferentes timeframes
timeframes_data = {
    'D1': daily_data,
    'H4': h4_data, 
    'H1': h1_data,
    'M15': m15_data
}

# Scoring multi-timeframe
results = scorer.score_multi_timeframe(timeframes_data, "US30")

# Buscar confluencia
confluence = scorer.find_confluence_setups(results)
if confluence:
    print(f"¡Confluencia detectada! Score promedio: {confluence['average_score']:.2%}")
```

## 🔧 Configuración Avanzada

### Parámetros de Feature Engineering

```python
from lightbgm.features import FeatureConfig

config = FeatureConfig(
    fib_lookback_period=100,     # Período para niveles Fibonacci
    macd_fast_period=12,         # MACD rápido
    macd_slow_period=26,         # MACD lento
    rsi_period=14,               # Período RSI
    volume_spike_threshold=2.0,  # Umbral picos de volumen
    # ... más parámetros
)
```

### Optimización Personalizada

```python
from lightbgm.optimization.optuna_optimizer import LightGBMOptunaOptimizer

optimizer = LightGBMOptunaOptimizer(
    n_trials=1000,        # Número de trials
    cv_folds=5,           # Cross-validation folds
    random_state=42       # Reproducibilidad
)

results = optimizer.optimize(data)
```

## 📊 Métricas y Validación

El sistema utiliza las siguientes métricas para optimización:

- **Métrica Principal**: AUC-ROC
- **Objetivo**: AUC > 0.80 en datos out-of-sample
- **Validación**: Time Series Cross-Validation
- **Confianza**: HIGH (>0.8), MEDIUM (0.6-0.8), LOW (<0.6)

### Análisis de Importancia de Features

```python
# Ver importancia de features del modelo entrenado
scorer = create_scorer_from_artifacts("models/optimization_results")
info = scorer.get_model_info()
print(info['feature_importance'])
```

## 🔍 Ejemplos Detallados

Ejecuta los ejemplos incluidos:

```bash
python examples/example_usage.py
```

Los ejemplos incluyen:
1. **Test rápido** con datos sintéticos
2. **Pipeline completo** con dataset pequeño
3. **Análisis de confluencia** multi-timeframe
4. **Simulación en tiempo real**
5. **Análisis de importancia** de features

## 📁 Estructura del Proyecto

```
lightbgm/
├── data/                    <- Datos históricos y procesados
├── models/                  <- Modelos entrenados y artefactos
├── notebooks/              <- Jupyter notebooks para análisis
├── examples/               <- Scripts de ejemplo
├── lightbgm/               <- Código fuente principal
│   ├── data_sources/       <- Generación de datos históricos
│   ├── indicators/         <- Indicadores técnicos personalizados
│   ├── optimization/       <- Pipeline de optimización Optuna
│   ├── scoring/           <- Sistema de scoring tiempo real
│   ├── modeling/          <- Entrenamiento y predicción
│   ├── features.py        <- Feature engineering dinámico
│   └── config.py          <- Configuración del proyecto
├── requirements.txt        <- Dependencias Python
└── README.md              <- Esta documentación
```

## 🎯 Casos de Uso

### 1. Desarrollo de Estrategias de Trading
```python
# Evaluar setups antes de entrar al mercado
result = scorer.score_setup(current_data, "US30", "H4")
if result.confidence == 'HIGH' and result.score > 0.8:
    print("¡Setup de alta probabilidad detectado!")
```

### 2. Análisis de Confluencia
```python
# Buscar alineación entre múltiples timeframes
confluence = scorer.find_confluence_setups(multi_tf_results)
if confluence and confluence['high_confidence_signals'] >= 2:
    print("Confluencia fuerte - considerar entrada")
```

### 3. Evaluación de Riesgo
```python
result = scorer.score_setup(data, "US100", "H1")
print(f"Evaluación de riesgo: {result.risk_assessment}")
print(f"Fuerza de señal: {result.signal_strength:.2%}")
```

## 🔮 Expansiones Futuras

El sistema está diseñado para integración con:

- **FinRL**: Reinforcement Learning para trading
- **Time-Series-Library**: Modelos Transformer avanzados
- **TensorTrade**: Simulación y backtesting
- **FinanceDatabase**: Datos fundamentales adicionales

## ⚠️ Consideraciones Importantes

- **GPU Requerida**: LightGBM está configurado para GPU por defecto
- **Datos en Tiempo Real**: Requiere configuración de broker/data feed
- **Optimización**: Puede tomar varias horas con 500+ trials
- **Memoria**: Datasets grandes requieren RAM suficiente

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una branch para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Reconocimientos

- [LightGBM](https://github.com/microsoft/LightGBM) - Framework de gradient boosting
- [Optuna](https://github.com/optuna/optuna) - Optimización automática de hiperparámetros
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - Estructura del proyecto
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Indicadores técnicos

---

**⚡ ¡Tu sistema de ML para trading está listo para generar señales de alta probabilidad!**

