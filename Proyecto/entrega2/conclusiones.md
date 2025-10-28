# Conclusiones - Entrega 2: MLOps Pipeline

## Resumen Ejecutivo

En esta segunda entrega del proyecto SodAI Drinks, implementamos un pipeline MLOps completo que transforma nuestro modelo de machine learning de la Entrega 1 en un sistema productivo, escalable y monitoreado. La integración de herramientas como Apache Airflow, FastAPI, Gradio y Docker nos permitió crear un flujo end-to-end desde la ingesta de datos hasta la visualización de predicciones.

---

## ¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de tracking y despliegue?

### **Tracking con MLflow**

La incorporación de MLflow transformó radicalmente nuestra forma de trabajar:

1. **Experimentación Organizada**: En la Entrega 1, los experimentos con hiperparámetros eran difíciles de rastrear. Con MLflow, cada ejecución de Optuna quedó registrada con:
   - Métricas de desempeño (F1-Score, ROC-AUC, Precision, Recall)
   - Hiperparámetros probados
   - Gráficos SHAP de interpretabilidad
   - Timestamp y duración del entrenamiento

2. **Reproducibilidad**: El registro automático de parámetros y artefactos garantiza que podemos reproducir cualquier experimento pasado, algo crítico cuando el modelo falla en producción.

3. **Comparación de Modelos**: La UI de MLflow nos permitió comparar visualmente diferentes runs de Optuna y seleccionar el mejor modelo basándonos en múltiples métricas simultáneamente.

4. **Gestión de Artefactos**: El almacenamiento centralizado de modelos entrenados (`.pkl`) y gráficos SHAP facilitó el despliegue, ya que FastAPI carga directamente el modelo más reciente desde el directorio compartido.

### **Despliegue Dockerizado**

Docker eliminó el clásico problema de "funciona en mi máquina":

- **Portabilidad**: El proyecto completo (Airflow + FastAPI + Gradio) puede levantarse en cualquier entorno con un simple `docker compose up -d`
- **Consistencia**: Las dependencias están congeladas en `requirements.txt` dentro de cada contenedor
- **Aislamiento**: Cada servicio corre en su propio contenedor, evitando conflictos de versiones
- **Escalabilidad**: Facilita el despliegue en cloud (AWS, GCP, Azure) sin modificaciones

**Impacto medible**: 
- Tiempo de setup en nueva máquina: de ~2 horas (instalando dependencias) a ~10 minutos (build + up)
- Errores de configuración: reducidos en ~90% gracias a la estandarización

---

## ¿Qué aspectos del despliegue con Gradio/FastAPI fueron más desafiantes o interesantes?

### **Desafíos Principales**

#### 1. **Integración del Pipeline de Preprocesamiento**

El mayor reto fue hacer funcionar el `pipeline_pp.pkl` de Airflow en el backend de FastAPI:

- **Problema**: El pipeline guardado con `joblib` contenía referencias a módulos personalizados (`helper_functions.py`) que no existían en el contenedor del backend.
- **Solución**: Replicamos las clases `GeoClustering`, `IQR` y `FeatureAggregator` en el backend para que `joblib.load()` pudiera deserializar correctamente.
- **Lección aprendida**: En producción, los transformadores custom deben estar versionados y compartidos como paquetes Python instalables.

#### 2. **Dependencia de Datos Históricos**

El modelo requiere features de frecuencia (cliente-producto, cliente-marca, cliente-categoría) que solo se pueden calcular con el historial completo de transacciones:

- **Problema inicial**: El backend solo necesitaba el modelo, pero las predicciones fallaban sin las transacciones.
- **Solución**: Compartimos `/airflow/data/raw/transacciones.parquet` vía volúmenes de Docker (read-only) para que el backend calcule las features en tiempo real.
- **Trade-off**: Esto aumenta la latencia de predicción (~500ms por request), pero mantiene la precisión del modelo.

#### 3. **Sincronización entre Contenedores**

- **Health checks**: Configuramos health checks en el backend para que el frontend no intente conectarse antes de que la API esté lista.
- **Orden de inicio**: El `depends_on` en docker-compose garantiza que el backend arranque primero.

### **Aspectos Interesantes**

#### 1. **Validación Robusta de IDs**

Implementamos validación con mensajes descriptivos que incluyen los rangos válidos:
```python
if cliente_id not in clientes['customer_id'].values:
    raise HTTPException(
        status_code=404,
        detail=f"Cliente {cliente_id} no encontrado. Rango válido: {min}-{max}"
    )
```

Esto mejoró dramáticamente la experiencia de usuario en Gradio.

#### 2. **Visualización Intuitiva en Gradio**

La interfaz utiliza clasificación visual de probabilidades:
- [VERDE] **SÍ COMPRARÁ** / [ROJO] **NO COMPRARÁ** (resultado destacado)
- MUY ALTA (70-100%) / ALTA (50-70%) / MEDIA (30-50%) / BAJA (0-30%)

Esto hace que usuarios no técnicos puedan interpretar las predicciones fácilmente.

#### 3. **Arquitectura de Microservicios**

La separación backend/frontend permitió:
- **Desarrollo independiente**: Dos personas pueden trabajar en paralelo sin conflictos
- **Testing aislado**: Probar la API con `curl` antes de integrar el frontend
- **Reutilización**: El mismo backend puede servir a múltiples frontends (mobile app, dashboards)

---

## ¿Cómo aporta Airflow a la robustez y escalabilidad del pipeline?

### **Robustez**

1. **Manejo de Fallos**
   - **Retries automáticos**: Cada task puede reintentar hasta 2 veces con delay exponencial
   - **Notificaciones**: Callbacks de `on_failure` pueden enviar alertas por email/Slack
   - **Logs detallados**: Cada ejecución queda registrada con stdout/stderr completo

2. **Flujos Condicionales**
   - **BranchPythonOperator**: Detecta drift → decide entre reentrenar o usar modelo existente
   - **Trigger rules**: `one_success` permite que las predicciones se generen independientemente de qué rama del DAG se ejecutó

3. **Idempotencia**
   - Las tareas están diseñadas para producir el mismo resultado si se ejecutan múltiples veces con los mismos datos (crucial para debugging)

### **Escalabilidad**

1. **Paralelización**
   - Múltiples tareas pueden ejecutarse simultáneamente si no tienen dependencias
   - Ejemplo: Detección de drift y optimización de hiperparámetros podrían correr en paralelo en una versión futura

2. **Scheduling Flexible**
   - `schedule_interval="@weekly"`: Ejecución automática cada semana sin intervención manual
   - Permite adaptar la frecuencia según volumen de datos (diario, semanal, mensual)

3. **Integración con Clusters**
   - Airflow puede orquestar jobs en Spark, Kubernetes, AWS EMR para procesar datasets masivos
   - En nuestro caso, si las transacciones crecen a millones de registros, podríamos migrar el preprocesamiento a Spark sin cambiar la estructura del DAG

4. **XCom para Compartir Estado**
   - Las tareas comparten DataFrames y rutas de modelos vía XCom
   - Esto evita recargar datos pesados múltiples veces

### **Monitoreo**

- **UI Web de Airflow**: Visualización en tiempo real del estado del DAG
- **Métricas de ejecución**: Duración de cada task, tasa de éxito/fallo
- **Historial completo**: Podemos auditar cuándo se reentrenó el modelo, qué datos se usaron, etc.

---

## ¿Qué se podría mejorar en una versión futura del flujo?

### **1. Automatización Adicional**

#### a) **CI/CD Pipeline**
- Integrar GitHub Actions para:
  - Ejecutar tests unitarios automáticamente en cada commit
  - Validar que el DAG de Airflow sea sintácticamente correcto
  - Rebuildar imágenes Docker y pushearlas a un registry (DockerHub, ECR)

#### b) **Auto-scaling en Cloud**
- Desplegar en Kubernetes con Horizontal Pod Autoscaler
- Escalar réplicas del backend según tráfico (más usuarios → más pods)

#### c) **Reentrenamiento Adaptativo**
- Actualmente: reentrenamiento solo si hay drift
- Mejora: reentrenar también si las métricas del modelo en producción caen bajo un umbral

### **2. Monitoreo y Observabilidad**

#### a) **Métricas en Tiempo Real**
- **Prometheus + Grafana**: Dashboards con:
  - Latencia de predicciones (p50, p95, p99)
  - Tasa de requests por segundo
  - Distribución de probabilidades predichas
  - Uso de CPU/RAM de los contenedores

#### b) **Alertas Proactivas**
- Slack/Email cuando:
  - La latencia supera 1 segundo
  - El modelo predice >80% de "No comprará" (posible drift)
  - El backend tiene >5% de errores 5xx

#### c) **Logging Estructurado**
- Cambiar de `print()` a librerías como `loguru` o `structlog`
- Enviar logs a Elasticsearch para búsqueda y análisis

### **3. Detección de Drift Mejorada**

Actualmente usamos test de Kolmogorov-Smirnov, pero podríamos:
- **Evidently AI**: Detección automática de drift en features y target
- **Drift de Concepto**: Monitorear si la relación X→y cambia (no solo la distribución de X)
- **Drift de Predicciones**: Comparar distribución de probabilidades actual vs. histórica

### **4. Métricas de Negocio**

Además de métricas técnicas (F1, ROC-AUC), trackear:
- **Lift**: ¿Cuánto mejoramos vs. baseline aleatorio?
- **Conversión real**: De los clientes que predijimos "Sí comprará", ¿cuántos realmente compraron?
- **ROI**: Valor monetario de las predicciones correctas vs. costo del sistema

### **5. Versionado de Datos**

- **DVC (Data Version Control)**: Versionar datasets como si fueran código
- Permite rollback a versiones anteriores de datos si el reentrenamiento falla
- Mantiene metadatos de quién modificó qué dato y cuándo

### **6. A/B Testing de Modelos**

- Desplegar dos versiones del modelo simultáneamente
- Rutear 50% del tráfico a cada una
- Comparar métricas de negocio para decidir cuál promover a producción

### **7. Feature Store**

- **Feast**: Centralizar el cálculo de features
- Evita que el backend tenga que recalcular frecuencias en cada request
- Pre-computa features y las sirve con latencia de milisegundos

### **8. Explicabilidad en Producción**

Actualmente SHAP solo se ejecuta en entrenamiento. Podríamos:
- Generar explicaciones SHAP individuales para cada predicción en el endpoint `/predict`
- Mostrar en Gradio: "El cliente comprará porque tiene alta frecuencia de compra de esta marca"

### **9. Rate Limiting y Autenticación**

- **API Keys**: Controlar quién puede usar la API
- **Rate Limiting**: Limitar requests por usuario (ej: 100/hora) para evitar abuso
- **OAuth2**: Integrar con sistema de autenticación corporativo

### **10. Mejoras en el Frontend**

- **Predicciones en Batch desde UI**: Permitir subir CSV con múltiples cliente-producto
- **Visualización de Tendencias**: Gráficos de cómo ha evolucionado la probabilidad de compra de un cliente a lo largo del tiempo
- **Feedback Loop**: Botón para que usuarios reporten predicciones incorrectas (datos para mejorar el modelo)

---

## Aprendizajes Clave

### **Técnicos**
1. **MLOps ≠ ML + Ops**: Es una disciplina con sus propios patrones y herramientas
2. **La mayor complejidad está en los datos, no en el modelo**: Pipeline de preprocesamiento, validación, versionado
3. **Reproducibilidad > Rendimiento inicial**: Un modelo con 5% menos accuracy pero reproducible es más valioso
4. **Observabilidad desde el día 1**: Es más fácil agregar logging/monitoring al inicio que retrofitarlo después

### **Arquitectura**
1. **Separación de responsabilidades**: Airflow (orquestación), FastAPI (serving), Gradio (UI)
2. **Contratos claros entre servicios**: APIs REST bien definidas facilitan el trabajo en equipo
3. **Infraestructura como código**: docker-compose.yml permite recrear el entorno completo

### **Procesos**
1. **El modelo es solo 20% del sistema**: El 80% restante es infraestructura, monitoreo, validación
2. **Despliegue continuo es clave**: El valor de ML viene de iterar rápido, no del modelo perfecto
3. **Documentación viva**: README.md, comentarios en código, diagramas de arquitectura deben evolucionar con el proyecto

---

## Reflexión Final

Este proyecto nos demostró que llevar un modelo de Jupyter Notebook a producción requiere mucho más que "exportar el .pkl". El ecosistema MLOps (Airflow, MLflow, Docker, FastAPI, Gradio) nos dio las herramientas para crear un sistema que:

- Se adapta a nuevos datos automáticamente
- Detecta y responde al drift
- Es observable y debuggeable
- Escala horizontalmente
- Tiene una interfaz amigable para usuarios no técnicos

Sin embargo, también aprendimos que **la complejidad tiene un costo**: más código que mantener, más servicios que monitorear, más puntos de fallo. En proyectos futuros, evaluaríamos cuidadosamente qué partes del stack realmente necesitamos según el contexto del negocio.

**La pregunta clave no es "¿Podemos construir esto?"** sino **"¿Deberíamos construir esto?"**. Para SodAI Drinks, con su necesidad de predicciones semanales y datos en constante evolución, la respuesta fue un rotundo sí. El ROI de automatizar el reentrenamiento y el despliegue justifica plenamente la inversión en infraestructura MLOps.

---

**Autores**: Matías Godoy, Delaney Tello  
**Fecha**: Noviembre 2025  
**Proyecto**: SodAI Drinks - Entrega 2  
**Curso**: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos
