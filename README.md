# Sistema de Recomendación con PySpark — Last.fm HetRec 2011

**Materia:** Big Data Analytics con Spark — Licenciatura en Data Science  
**Modelos:** ALS (PySpark) + BPR (PyTorch) con comparación empírica  
**Dataset:** Last.fm HetRec 2011 (versión 2k, GroupLens)  
**Entorno:** Google Colab (gratuito)

---

## Estructura del repositorio

```
.
├── TP_Recomendacion_LastFM.ipynb   # Notebook principal ejecutado con outputs
├── informe.pdf                     # Metodología, justificación y escalabilidad
└── README.md                       # Este archivo
```

---

## Cómo reproducir

### 1. Entorno

Ejecutar en **Google Colab gratuito**. No se requiere GPU — ALS corre íntegramente en Spark y BPR es liviano en CPU.

### 2. Descargar el dataset

El notebook descarga el dataset automáticamente en la celda 0.4 con:

```bash
wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
```

No es necesario subir archivos manualmente.

### 3. Ejecutar el notebook

1. Abrir `TP_Recomendacion_LastFM.ipynb` en Google Colab
2. **Reiniciar el kernel** (`Runtime > Restart runtime`)
3. Ejecutar todas las celdas en orden (`Runtime > Run all`)

El notebook corre de punta a punta sin intervención manual, **excepto la Sección 9** (bonus de recomendaciones personalizadas), que requiere ingresar una API key de Last.fm por `getpass`. Para obtener una API key gratuita: [last.fm/api/account/create](https://www.last.fm/api/account/create)

Para saltear la Sección 9, simplemente no ejecutar las celdas 9.1–9.6.

### 4. Dependencias

PySpark se instala automáticamente en la primera celda:

```bash
pip install pyspark
```

El resto de las librerías (`torch`, `sklearn`, `matplotlib`, `pandas`, `numpy`) vienen preinstaladas en Colab.

---

## Resumen del pipeline

| Etapa | Herramienta |
|-------|------------|
| Descarga y carga | PySpark |
| EDA y exploración | PySpark |
| Filtrado del subconjunto | PySpark |
| Feature engineering | PySpark |
| Re-indexación | PySpark `StringIndexer` |
| Persistencia entre celdas | Parquet (DBFS-compatible) |
| Modelo principal | PySpark `ALS` (implicitPrefs) |
| Modelo de comparación | PyTorch `BPR` |
| Evaluación | Precision@10, Recall@10, NDCG@10 |
| Visualización de embeddings | PCA (sklearn) |

---

## Dataset

- **Nombre:** Last.fm HetRec 2011 (hetrec2011-lastfm-2k)
- **Fuente:** [grouplens.org/datasets/hetrec-2011](https://grouplens.org/datasets/hetrec-2011/)
- **Tamaño completo:** 1.892 usuarios, 17.632 artistas, 92.834 interacciones
- **Subconjunto usado:** usuarios con 50–500 artistas únicos escuchados + artistas con ≥5 oyentes únicos → 1.827 usuarios, 2.828 artistas, 70.454 interacciones
- **Tipo de interacción:** implícita (play counts, no ratings)
