# Conclusiones - Entrega 2: MLOps Pipeline

## ¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de tracking y despliegue?

MLflow transformó la experimentación al registrar automáticamente cada run de Optuna con sus métricas, hiperparámetros y artefactos. Esto permitió reproducibilidad completa y comparación visual de modelos, algo que era caótico en la Entrega 1. El almacenamiento centralizado facilitó que FastAPI cargue directamente el modelo más reciente sin configuración manual.

Docker eliminó el problema de "funciona en mi máquina". El proyecto completo se levanta en cualquier entorno con `docker compose up -d`, reduciendo el tiempo de setup de 2 horas a 10 minutos y los errores de configuración en un 90%. La containerización también facilita el despliegue en cloud sin modificaciones.

## ¿Qué aspectos del despliegue con Gradio/FastAPI fueron más desafiantes o interesantes?

El mayor desafío fue integrar el pipeline de preprocesamiento. El `pipeline_pp.pkl` guardado por Airflow contenía referencias a clases custom que inicialmente no existían en el contenedor del backend. Esto nos enseñó que en producción los transformadores deben estar versionados como paquetes Python instalables.

Otro reto fue la dependencia de datos históricos. El modelo necesita calcular features de frecuencia usando el historial completo de transacciones, por lo que compartimos los parquets vía volúmenes Docker. Esto aumenta la latencia (~500ms por request) pero mantiene la precisión del modelo.

Lo más interesante fue implementar validación robusta de IDs con mensajes descriptivos que incluyen rangos válidos, mejorando dramáticamente la experiencia de usuario. La separación backend/frontend permitió desarrollo paralelo, testing aislado y reutilización del mismo backend para múltiples interfaces.

## ¿Cómo aporta Airflow a la robustez y escalabilidad del pipeline?

Airflow proporciona robustez mediante retries automáticos, flujos condicionales (BranchPythonOperator para decidir entre reentrenar o usar el modelo existente según drift) y logs detallados de cada ejecución. La idempotencia de las tareas permite debugging sin efectos secundarios.

En escalabilidad, Airflow permite paralelización de tareas independientes, scheduling flexible (`@weekly` en nuestro caso) y XCom para compartir estado entre tareas. Su capacidad de integrarse con clusters (Spark, Kubernetes) significa que si nuestros datos crecen a millones de registros, podemos migrar el preprocesamiento a Spark sin cambiar la estructura del DAG. La UI web ofrece monitoreo en tiempo real y auditoría completa del historial de ejecuciones.

## ¿Qué se podría mejorar en una versión futura del flujo?

**Automatización**: Implementar CI/CD con GitHub Actions para tests automáticos y validación del DAG, desplegar en Kubernetes con auto-scaling, y agregar reentrenamiento adaptativo cuando las métricas en producción caigan bajo un umbral.

**Monitoreo**: Integrar Prometheus + Grafana para métricas en tiempo real (latencia, tasa de requests, distribución de probabilidades), alertas proactivas por Slack/Email, y logging estructurado con Elasticsearch.

**Drift y Métricas**: Mejorar la detección de drift con Evidently AI (drift de concepto y predicciones, no solo distribución), trackear métricas de negocio (lift, conversión real, ROI) y versionar datos con DVC.

**Performance**: Implementar un Feature Store (Feast) para pre-computar features y reducir latencia de 500ms a milisegundos, y agregar explicabilidad SHAP individual en cada predicción.

**Seguridad**: Rate limiting, API keys y OAuth2 para autenticación corporativa.

## Reflexión Final

Este proyecto demostró que llevar un modelo a producción requiere mucho más que "exportar el .pkl". El ecosistema MLOps nos permitió crear un sistema que se adapta automáticamente a nuevos datos, detecta drift, es observable y escala horizontalmente. 

Aprendimos que la mayor complejidad está en los datos y la infraestructura (80%), no en el modelo (20%). La reproducibilidad es más valiosa que el rendimiento inicial, y la observabilidad debe implementarse desde el día 1. La separación de responsabilidades (Airflow para orquestación, FastAPI para serving, Gradio para UI) facilitó el desarrollo paralelo y el testing aislado.

Sin embargo, la complejidad tiene un costo: más código que mantener, más servicios que monitorear, más puntos de fallo. En proyectos futuros evaluaríamos cuidadosamente qué partes del stack realmente necesitamos según el contexto del negocio. Para SodAI Drinks, con predicciones semanales y datos en constante evolución, el ROI de automatizar el reentrenamiento justifica plenamente la inversión en infraestructura MLOps.

---

**Autores**: Matías Godoy, Delaney Tello  
**Fecha**: Noviembre 2025  
**Proyecto**: SodAI Drinks - Entrega 2  
**Curso**: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos
