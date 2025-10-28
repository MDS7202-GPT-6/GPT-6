"""
Módulo de compatibilidad para cargar el pipeline_pp.pkl de Airflow.
El pipeline fue serializado con referencias a helper_functions, por lo que necesitamos
este módulo con las mismas clases.
"""

from transformers import GeoClustering, IQR, FeatureAggregator

__all__ = ['GeoClustering', 'IQR', 'FeatureAggregator']
