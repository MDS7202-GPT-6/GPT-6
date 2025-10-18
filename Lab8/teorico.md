# Respuestas teóricas Lab 8 - Docker

### **1. ¿Cómo se diferencia Docker de una máquina virtual (VM)?**  
Docker usa contenedores ligeros que comparten el mismo sistema operativo del host, mientras que una VM incluye un sistema operativo completo, lo que la hace más pesada y lenta.

---

### **2. ¿Cuál es la diferencia entre usar Docker y ejecutar la aplicación directamente en el sistema local?**  
Docker aísla la aplicación y sus dependencias en un entorno controlado y reproducible; ejecutarla localmente depende del sistema anfitrión, lo que puede causar incompatibilidades.

---

### **3. ¿Cómo asegura Docker la consistencia entre diferentes entornos de desarrollo y producción?**  
Docker usa imágenes con versiones fijas del sistema base y librerías, garantizando que el entorno sea idéntico en cualquier máquina o servidor.

---

### **4. ¿Cómo se gestionan los volúmenes en Docker para la persistencia de datos?**  
Los volúmenes almacenan datos fuera del contenedor, permitiendo que persistan incluso si el contenedor se detiene o elimina.

---

### **5. ¿Qué son Dockerfile y docker-compose.yml, y cuál es su propósito?**  
- **Dockerfile:** define los pasos para construir una imagen (instalaciones, copias, puertos, comandos).  
- **docker-compose.yml:** orquesta varios contenedores y servicios (API, base de datos, etc.) de forma conjunta.