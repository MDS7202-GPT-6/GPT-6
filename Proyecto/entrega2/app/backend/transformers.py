"""
Clases transformadoras personalizadas para el pipeline de preprocesamiento.
Deben ser idénticas a las definidas en helper_functions.py de Airflow.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


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
    """Eliminación de outliers usando método IQR - reemplaza con NaN, no elimina filas"""
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
            # Reemplazar outliers con NaN en lugar de eliminar filas
            mask = (X[col] < self.inferior[col]) | (X[col] > self.superior[col])
            X.loc[mask, col] = np.nan
        return X


class FeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Genera features de frecuencia, trimestre y otras agregaciones temporales.
    Requiere df_transacciones como parámetro en transform().
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, df_transacciones=None):
        if df_transacciones is None:
            raise ValueError("Necesitas pasar df_transacciones a transform()")

        X = X.copy()
        df_transacciones = df_transacciones.copy()
        df_transacciones["purchase_date"] = pd.to_datetime(df_transacciones["purchase_date"])

        # Crear fecha_actual para cada fila
        X["fecha_actual"] = pd.to_datetime(
            X["Año"].astype(str) + "-W" + X["Semana"].astype(str) + "-1",
            format="%G-W%V-%u"
        )
        
        # Preparar transacciones solo con columnas necesarias para evitar duplicados
        trans = df_transacciones[["customer_id", "product_id", "purchase_date"]].copy()
        trans = trans.rename(columns={"purchase_date": "fecha_compra"})

        # Merge para obtener compras previas
        merged = X.merge(trans, on=["customer_id", "product_id"], how="left")
        merged = merged[merged["fecha_compra"] < merged["fecha_actual"]]

        # Frecuencia por producto
        freq = (
            merged.groupby(["customer_id", "product_id", "Año", "Semana"])
            .size()
            .reset_index(name="frecuencia")
        )
        X = X.merge(freq, on=["customer_id", "product_id", "Año", "Semana"], how="left")
        X["frecuencia"] = X["frecuencia"].fillna(0).astype(int)

        # Trimestre
        X["trimestre"] = X["fecha_actual"].dt.quarter

        # Frecuencia por categoría
        if "category" in X.columns:
            freq_cat = (
                merged.groupby(["customer_id", "category", "Año", "Semana"])
                .size()
                .reset_index(name="frecuencia_categoria")
            )
            X = X.merge(freq_cat, on=["customer_id", "category", "Año", "Semana"], how="left")
            X["frecuencia_categoria"] = X["frecuencia_categoria"].fillna(0).astype(int)

        # Frecuencia por marca
        if "brand" in X.columns:
            freq_brand = (
                merged.groupby(["customer_id", "brand", "Año", "Semana"])
                .size()
                .reset_index(name="frecuencia_brand")
            )
            X = X.merge(freq_brand, on=["customer_id", "brand", "Año", "Semana"], how="left")
            X["frecuencia_brand"] = X["frecuencia_brand"].fillna(0).astype(int)

        X = X.drop(columns=["fecha_actual"])
        return X
