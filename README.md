# Sistema de Recomendación — Last.fm 

---

## Estructura del repositorio

```
.
├── TP_Recomendacion_LastFM.ipynb   # Notebook ejecutado con los outputs
├── informe.pdf                     # Explicación y respuestas de la consigna
└── README.md                       # Este archivo
```

---

## Cómo reproducir

### 1. Entorno

Ejecutar en **Google Colab**

### 2. Descargar el dataset

El notebook descarga el dataset automáticamente en la celda 0.4 con:

```bash
wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
```

No es necesario subir los archivos manualmente.

### 3. Ejecutar el notebook

1. Ejecutar todas las celdas en orden (`Run all`)

El notebook corre de punta a punta sin intervención manual, **excepto en la Sección 9** (bonus de recomendaciones personalizadas), que requiere ingresar una API key de Last.fm por `getpass`. Para obtener una API key gratuita: [last.fm/api/account/create](https://www.last.fm/api/account/create)

Para saltear la Sección 9, simplemente no ejecutar las celdas 9.1–9.6.

O sino, usar la API Key del informe.

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
