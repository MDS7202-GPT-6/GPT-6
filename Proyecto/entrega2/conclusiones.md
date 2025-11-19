# Conclusiones - Entrega 2: MLOps Pipeline

## ¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de tracking y despliegue?

MLflow transformó la experimentación al registrar automáticamente cada run de Optuna con sus métricas, hiperparámetros y artefactos. Esto permitió reproducibilidad completa y comparación visual de modelos, algo que era caótico en la Entrega 1.

Por su parte, Airflow ayuda a visualizar correctamente la ejecución del pipeline, facilitando la identificación de fallos y cuellos de botella. Consideramos que esta es una de las herramientas más útiles del curso.

Docker, por otro lado, simplificó enormemente el despliegue. Al contenerizar la aplicación con FastAPI y Gradio, garantizamos que el entorno de producción fuera idéntico al de desarrollo, eliminando problemas de dependencias y versiones. Además, la orquestación con Docker Compose permitió levantar todo el stack (Airflow, MLflow, backend, frontend) con un solo comando. Creemos con firmeza que esto ha sido lo más valioso a aprender en el curso, ya que se que esto se usa en la industria.

## ¿Qué aspectos del despliegue con Gradio/FastAPI fueron más desafiantes o interesantes?

El mayor desafío fue integrar el pipeline de preprocesamiento. El `pipeline_pp.pkl` guardado por Airflow contenía referencias a clases custom que inicialmente no existían en el contenedor del backend. 

Lo más interesante fue implementar la predicción en tiempo real con FastAPI y manejo eficiente de concurrencia. La documentación automática con Swagger facilitó la prueba de endpoints. Creemos que conocer aunque sea superficialmente FastAPI es muy útil para nuestro futuro profesional, ya que es una de las herramientas más usadas en la industria para levantar modelos ML.

## ¿Cómo aporta Airflow a la robustez y escalabilidad del pipeline?

Airflow proporciona robustez mediante retries automáticos, flujos condicionales, como con el caso del drift o poder definir dependencias claras entre tareas. Si una tarea falla, Airflow puede reintentarla sin afectar el resto del pipeline.

En escalabilidad, Airflow permite paralelización de tareas independientes y scheduling flexible. Su capacidad de integrarse con clusters (Spark, Kubernetes) significa que si nuestros datos crecen a millones de registros, podemos migrar el preprocesamiento a Spark sin cambiar la estructura del DAG. La UI web ofrece monitoreo en tiempo real y una revisión completa del historial de ejecuciones.

## ¿Qué se podría mejorar en una versión futura del flujo? ¿Qué partes automatizarían más, qué monitorearían o qué métricas agregarían?

Para futuras versiones, sería valioso poder ejecutar más entrenamientos en paralelo, pudiendo monitorear múltiples experimentos simultáneamente. 

Otra mejora podría ser que se quede constantemente ejecutando el pipeline en un servidor esperando un drift, para no tener que estar ejecutándolo manualmente cada vez o tenerlo programado cada cierta cantidad de días, y que al momento de detectar drift se ejecute automáticamente. De esta forma se tendría un sistema más autónomo/automatizado.

## Reflexión Final

Este proyecto demostró que llevar un modelo a producción requiere mucho más que exportar el archivo pkl. El ecosistema MLOps nos permitió crear un sistema que se adapta automáticamente a nuevos datos, detecta drift, es observable y escala horizontalmente. 

Aprendimos que la mayor complejidad está en los datos y la infraestructura, no en el modelo. Consideramos que aprendimos cosas muy valiosas para la industria en este proyecto, especialmente en lo que respecta a Airflow y Docker. 

Sentimos que realmente estamos aplicando todo lo aprendido en la Universidad con un proyecto más similar a lo que nos enfrentaríamos en el mundo laboral.

---

**Autores**: Matías Godoy, Delaney Tello  
**Fecha**: Noviembre 2025  
**Curso**: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos
