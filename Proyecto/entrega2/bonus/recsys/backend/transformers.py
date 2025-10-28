"""
Transformadores personalizados para compatibilidad con el pipeline de Airflow.
Estos deben coincidir exactamente con los definidos en airflow/dags/helper_functions.py
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np


class GeoClustering(BaseEstimator, TransformerMixin):
    """Clustering geográfico basado en coordenadas X, Y"""
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self, X, y=None):
        if "X" in X.columns and "Y" in X.columns:
            coords = X[["X", "Y"]].dropna()
            if len(coords) > 0:
                self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                self.kmeans.fit(coords)
        return self

    def transform(self, X):
        X = X.copy()
        if self.kmeans is not None and "X" in X.columns and "Y" in X.columns:
            mask = X[["X", "Y"]].notna().all(axis=1)
            X.loc[mask, "geo_cluster"] = self.kmeans.predict(X.loc[mask, ["X", "Y"]])
            X["geo_cluster"] = X["geo_cluster"].fillna(-1).astype(int)
        else:
            X["geo_cluster"] = -1
        return X


class IQR(BaseEstimator, TransformerMixin):
    """Eliminación de outliers usando método IQR"""
    def __init__(self, l=1.5):
        self.l = l
        self.inferior = None
        self.superior = None

    def fit(self, X, y=None):
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.inferior = q1 - self.l * iqr
        self.superior = q3 + self.l * iqr
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            mask = (X[col] < self.inferior[col]) | (X[col] > self.superior[col])
            X.loc[mask, col] = np.nan
        return X


class FeatureAggregator(BaseEstimator, TransformerMixin):
    """Agregador de features personalizadas"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, df_transacciones=None):
        X = X.copy()
        
        if df_transacciones is not None:
            # Frecuencias cliente-producto
            freq_cp = df_transacciones.groupby(['customer_id', 'product_id']).size().reset_index(name='frecuencia')
            X = X.merge(freq_cp, on=['customer_id', 'product_id'], how='left')
            X['frecuencia'] = X['frecuencia'].fillna(0)
            
            # Frecuencias cliente-categoría
            trans_cat = df_transacciones.merge(
                X[['product_id', 'category']].drop_duplicates(), 
                on='product_id', 
                how='left'
            )
            freq_cat = trans_cat.groupby(['customer_id', 'category']).size().reset_index(name='frecuencia_categoria')
            X = X.merge(freq_cat, on=['customer_id', 'category'], how='left')
            X['frecuencia_categoria'] = X['frecuencia_categoria'].fillna(0)
            
            # Frecuencias cliente-brand
            trans_brand = df_transacciones.merge(
                X[['product_id', 'brand']].drop_duplicates(), 
                on='product_id', 
                how='left'
            )
            freq_brand = trans_brand.groupby(['customer_id', 'brand']).size().reset_index(name='frecuencia_brand')
            X = X.merge(freq_brand, on=['customer_id', 'brand'], how='left')
            X['frecuencia_brand'] = X['frecuencia_brand'].fillna(0)
        
        # Crear features temporales
        if 'Semana' in X.columns:
            X['trimestre'] = ((X['Semana'] - 1) // 13) % 4 + 1
        
        return X
