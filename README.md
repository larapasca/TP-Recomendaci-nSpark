# Sistema de Recomendación — Last.fm 360K

**Grupo:** Lara Pascaretta (64776), Agustina Kohan Miller (64231), Stefania Ranucci (63694)  
---

## Estructura del repositorio

```
.
├── tp-recomendacion-spark.ipynb    # Notebook ejecutado con outputs visibles
├── informe.pdf                     # Informe: metodología, análisis, escalabilidad
└── README.md                       # Este archivo
```

---

## Resumen ejecutivo

Sistema de recomendación de artistas musicales usando **Alternating Least Squares (ALS)** en PySpark, entrenado sobre Last.fm 360K (17.5M interacciones). El modelo alcanza **Precision@10=0.0651** y **NDCG@10=0.0824** sobre un subconjunto filtrado de 22k usuarios y 25k artistas.

**Decisiones clave:**
- Dataset: Last.fm 360K (implicit feedback: play counts, no ratings)
- Modelo: ALS con `implicitPrefs=True` (factorización matricial nativa en Spark)
- Subconjunto: usuarios con 10–500 reproducciones totales, artistas con ≥50 oyentes
- Comparación: ALS vs BPR (ALS ~29% mejor en Precision@10)
- Extras: visualización PCA de embeddings, recomendaciones personalizadas vía API

---

## Cómo reproducir

### 1. Entorno

**Recomendado:** Google Colab gratuito o Kaggle Notebooks (ambos funcionan igual).

Por qué no Databricks Community Edition:
- Solo ofrece Serverless, que bloquea `pyspark.ml.recommendation.ALS`
- El trabajo requiere entrenar ALS nativo, sin workarounds

### 2. Preparación

1. Abrir el notebook en Colab o Kaggle
2. **El dataset se descarga automáticamente** en la celda 1.4
   - Descarga desde Internet Archive (mirror oficial + confiable)
   - ~540 MB, tarda ~30 segundos
   - Se extrae a `/tmp/lastfm-dataset-360K/`

No es necesario descargar manualmente nada.

### 3. Ejecutar

```
Menú → Run all cells
```

El notebook corre **de punta a punta sin intervención manual** (~30–45 min en Colab), **excepto**:

#### Sección 9: Recomendaciones personalizadas (opcional)

Si querés activarla:
1. Cambiar `RUN_LASTFM_BONUS = False` → `RUN_LASTFM_BONUS = True` en celda 9.0
2. Crear una API key gratuita: [last.fm/api/account/create](https://www.last.fm/api/account/create)
3. La celda 9.1 pedirá que ingreses la key por `getpass()`

Si no la querés: Simplemente saltear celdas 9.1–9.6 (todo funciona igual).

### 4. Dependencias

Se instalan automáticamente en la primera celda:

```bash
pip install pyspark
```

Otras librerías vienen pre-instaladas en Colab:
- pandas, numpy, matplotlib
- scikit-learn (para PCA)
- torch (para BPR)
- requests (para API Last.fm)

---

## Descripción del pipeline

| Etapa | Herramienta | Entrada | Salida |
|-------|-------------|---------|--------|
| **Carga** | PySpark SQL | `.tsv` (gz) | DataFrame 17.5M filas |
| **EDA** | PySpark + matplotlib | DataFrame crudo | Histogramas, densidad |
| **Filtrado** | PySpark SQL | 17.5M → 809k interacciones | Dataset sub-muestreado |
| **Features** | PySpark SQL | Usuario + artista | log1p(popularidad), conteos |
| **Indexación** | StringIndexer | IDs textuales | Índices Int (0–22k usuarios, 0–25k artistas) |
| **Split** | Window partitioning | 809k interacciones | 80% train, 20% test **por usuario** |
| **Entrenamiento** | PySpark ALS | train_df + hyperparams | Embeddings usuario (50-dim) + artista (50-dim) |
| **Evaluación** | Spark SQL + UDFs | test_df + embeddings | Precision@10, Recall@10, NDCG@10 |
| **PCA** | scikit-learn | Artist embeddings | 2D projection + gráfico |
| **Comparación** | PyTorch + Spark | Mismo split | BPR (emb_dim=32, 25 épocas) |

---

## Documentación del subconjunto y sesgos

### Filtrado

- **Usuarios:** total_plays ∈ [10, 500] → retiene **6.2%** (22,125 / 359,349)
  - Excluye: usuarios sin historial (< 10 plays) y power users extremos (> 500 plays)
- **Artistas:** unique_listeners ≥ 50 → retiene **15.7%** (25,128 / 160,168)
  - Excluye: artistas de tail muy largo (< 50 oyentes, ruido colaborativo)

### Consecuencias

- **Resultado:** 809,540 interacciones, densidad 0.15% (vs. 0.03% completo)
- **Sesgo:** Las métricas @K son **confiables solo para catálogo popular**
  - No miden recomendación de cola larga
  - No miden cold start (usuarios nuevos)
  - El modelo probablemente mejora diversidad con usuarios extremos

**Mitigación en el trabajo:** Se documenta y se reporta cómo escalaría con datos completos (informe, sección 6).

---

## Decisiones técnicas clave

### Por qué ALS (y no Two-Tower / BPR)

1. **Nativo en Spark:** Corre 100% en SQL/RDD, sin PyTorch → ideal para Colab sin GPU garantizada
2. **Implicit feedback:** Last.fm da play counts, no ratings → ALS con `implicitPrefs=True` está diseñado para esto
3. **Escalabilidad:** Factorización matricial distribuida → manejable en un nodo, trivial de escalar a cluster
4. **Rapidez:** Converge en ~6 iteraciones (< 2 min por entrenamiento)

### Por qué NO Two-Tower

- Requiere GPU o distribuido (PyTorch)
- Features adicionales (género, país) no son tan aprovechadas en este dataset
- ALS supera a BPR (alternativa más cercana) en todas las métricas

### Configuración de SparkSession

```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .getOrCreate()
```

- `driver.memory=4g`: Límite de Colab, suficiente para 809k filas
- `shuffle.partitions=200`: Ajustado para single node, reduce memory pressure
- `autoBroadcastJoinThreshold=-1`: Desactiva broadcasts (evita OOM en joins)

### Negative sampling

- **Estrategia:** ALS `implicitPrefs=True` trata no-observados como negativos implícitos
- **Ventaja:** Evita crossJoin (que generaría 57B pares usuario-artista)
- **Alternativa escalable:** Negative sampling aleatorio por partición (documentado en sección 6 del informe)

---

## Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| 1. Setup | Instalación PySpark, descarga automática dataset, imports |
| 2. EDA | Estadísticas, distribuciones, análisis de densidad |
| 3. Filtrado | Criterios de muestreo, cálculo de % retención |
| 4. Preprocesamiento | Features, StringIndexer, persistencia Parquet, train/test split |
| 5. Benchmark | Tiempos de cada etapa (cuello de botella: StringIndexer) |
| 6. Entrenamiento ALS | Grid search, selección de hyperparams, curva de convergencia |
| 7. Evaluación | Ranking-aware metrics, recomendaciones para 3 usuarios |
| 8. PCA | Visualización de embeddings en 2D |
| 9. Bonus | Recomendaciones personalizadas vía API (opcional) |
| 10. BPR | Comparación empírica con otro modelo |

---

## Escalabilidad a 100M interacciones

*(Documentado en el informe, sección 6)*

### Negative sampling

A esta escala, evitar crossJoin es crítico. Estrategia:
- Generate ~10 negativos por positivo (en lugar de todos los no-vistos)
- Muestreo por popularidad: artistas populares → más prob. ser negativo
- Hard negatives: artistas similares al positivo (usando embeddings)

### Almacenamiento y persistencia

- Los datos deben permanecer distribuidos (no `toPandas()`)
- Formatos: Parquet → Delta Lake (time travel, ACID)
- Cache inteligente: solo train_df, no validación completa en memoria

### Entrenamiento distribuido

Con TorchDistributor en Databricks (cluster 4 workers + GPU):
```python
torch_distributor = TorchDistributor(
    local_mode=False,
    num_processes=4
)
torch_distributor.run(train_bpr_distributed, train_data)
```

### Serving en tiempo real

Precomputar top-K por usuario, actualizar cada 24h:
- **Opción 1 (simple):** Delta Lake table + predicate pushdown
- **Opción 2 (baja latencia):** Redis (index → top 100 artistas)
- **Opción 3 (dinámico):** Buscar embedding del usuario en índice ANN (FAISS), re-rank con señales recientes

