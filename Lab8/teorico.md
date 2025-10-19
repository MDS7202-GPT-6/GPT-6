# Respuestas teóricas Lab 8 - Docker

### **1. ¿Cómo se diferencia Docker de una máquina virtual (VM)?**  
Docker usa contenedores ligeros que comparten el mismo sistema operativo del host y comparten sus recursos, mientras que una VM incluye un sistema operativo completo, lo que la hace más pesada y lenta.

Una VM  implica instalar y configurar un sistema operativo completo, en cambio Dockerse enfoca en empaquetar únicamente lo necesario para ejecutar la aplicación.

---

### **2. ¿Cuál es la diferencia entre usar Docker y ejecutar la aplicación directamente en el sistema local?**  
Docker aísla la aplicación y sus dependencias en un entorno controlado y reproducible; ejecutarla localmente depende del sistema operativo anfitrión  y del Hardware, lo que puede causar incompatibilidades al momento de desplegar en otro entorno.

---

### **3. ¿Cómo asegura Docker la consistencia entre diferentes entornos de desarrollo y producción?**  
Docker usa imágenes con versiones fijas del sistema base y librerías, garantizando que el entorno sea idéntico en cualquier máquina o servidor.

Estas imágenes son archivos inmutables que contienen todo lo necesario para ejecutar la aplicación, eliminando problemas de "funciona en mi máquina".

---

### **4. ¿Cómo se gestionan los volúmenes en Docker para la persistencia de datos?**  
Los volúmenes permiten guardar datos fuera del contenedor para que no se pierdan al eliminarlo o recrearlo.  Pueden ser creados y administrados por Docker o vincular carpetas locales del host. Así, los archivos del contenedor se guardan en el sistema local, asegurando persistencia y compatibilidad entre entornos.

Entonces, si por ejemplo quiero guardar una base de datos, puedo mapear el directorio de datos del contenedor a una carpeta en mi máquina local.

---

### **5. ¿Qué son Dockerfile y docker-compose.yml, y cuál es su propósito?**  

- **Dockerfile:** define los pasos para construir imagenes (instalaciones, copias, puertos, comandos).  
- **docker-compose.yml:** junta varios contenedores y servicios (API, base de datos, etc.) de forma conjunta, es decir, levanta varios contenedores con una sola instrucción.