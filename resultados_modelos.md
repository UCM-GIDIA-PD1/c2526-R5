# Regresión de retraso por horizonte temporal — Resultados

---

## delay_30m (scheduled_time_to_end >= 1800s)

### Resultados finales (test = enero 2026)

| Modelo | MAE (s) | MAE (min) | RMSE (s) | R² |
|---|---|---|---|---|
| Baseline media | 220.04 | 3.67 | 353.53 | -0.000 |
| Baseline persistencia | 159.77 | 2.66 | 273.04 | 0.4035 |
| **LGBM** | **134.28** | **2.24** | **251.99** | **0.5751** |
| **MLP** | pdte | pdte | pdte | pdte |

### Búsqueda de hiperparámetros (validación = oct–dic 2025)

| Estrategia | MAE val (s) | MAE val (min) | RMSE (s) | R² |
|---|---|---|---|---|
| LGBM Optuna (50 runs) | 131.21 | 2.19 | — | — |
| LGBM Random Search (20 runs) | 131.20 | 2.19 | — | — |
| MLP Optuna (30 runs) | 133.42 | 2.22 | — | — |
| MLP Random Search (30 runs) | 136.80 | 2.28 | 231.30 | 0.5722 |

### Mejores hiperparámetros delay_30m

**LGBM Optuna:**
- objective: regression_l1
- num_leaves: 511, max_depth: 16
- learning_rate: 0.05, min_child_samples: 100
- feature_fraction: 0.7426, bagging_fraction: 0.8165
- reg_alpha: 1.5346, reg_lambda: 1.2927, min_split_gain: 0.3704

**MLP Optuna (mejor trial):**
- hidden_layers: [256, 512, 128]
- dropout: 0.118
- learning_rate: 0.00238
- weight_decay: 4.45e-5
- batch_size: 8192

**MLP Random Search (trial 10):**
- hidden_layers: [512, 512]
- dropout: 0.454
- learning_rate: 0.00193
- weight_decay: 2.68e-6
- batch_size: 2048

---

## delay_end (scheduled_time_to_end < 1800s)

### Resultados finales (test = enero 2026)

| Modelo | MAE (s) | MAE (min) | RMSE (s) | R² |
|---|---|---|---|---|
| Baseline media | 231.81 | 3.86 | 398.87 | -0.001 |
| Baseline persistencia | 155.81 | 2.60 | 296.28 | 0.453 |
| **LGBM** | **108.98** | **1.82** | pdte | pdte |
| **MLP** | pdte | pdte | pdte | pdte |

### Búsqueda de hiperparámetros (validación = oct–dic 2025)

| Estrategia | MAE val (s) | MAE val (min) | RMSE (s) | R² |
|---|---|---|---|---|
| LGBM Optuna (50 runs) | 110.73 | 1.85 | — | — |
| LGBM Random Search (15 runs) | 111.00 | 1.85 | — | — |
| MLP Optuna (30 runs) | 120.15 | 2.00 | 228.10 | 0.6711 |
| MLP Random Search (30 runs) | 125.79 | 2.10 | 233.85 | 0.6544 |

### Mejores hiperparámetros delay_end

**LGBM Optuna:**
- objective: regression_l1
- num_leaves: 511, max_depth: -1
- learning_rate: 0.1, min_child_samples: 200
- feature_fraction: 0.9248, bagging_fraction: 0.7040
- reg_alpha: 1.5123, reg_lambda: 0.9404, min_split_gain: 0.4288

**MLP Optuna (trial 3):**
- hidden_layers: [512, 128]
- dropout: 0.317
- learning_rate: 0.000191
- weight_decay: 2.55e-4
- batch_size: 2048

**MLP Random Search (trial 1):**
- hidden_layers: [512, 64, 64, 64]
- dropout: 0.187
- learning_rate: 0.00102
- weight_decay: 1.2e-6
- batch_size: 2048
