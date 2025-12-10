Actúa como un arquitecto de software experto, documentador técnico y editor de código.

Tu tarea es leer y analizar exhaustivamente todo el contenido del repositorio o carpeta de trabajo actual, incluyendo todos los archivos de código fuente, configuración y documentación.

Una vez que domines completamente la lógica, la arquitectura y el flujo de trabajo del sistema, debes actualizar el archivo **README.md** existente con documentación técnica avanzada y diagramas **Mermaid**.

**Instrucciones Específicas para la Actualización del README.md:**

1.  **Posicionamiento:** Inserta cada sección *antes* de cualquier otra sección de alto nivel (como "Instalación" o "Uso") para priorizar la comprensión arquitectónica, pero *después* de la descripción principal del proyecto.
2.  **Formato de Diagrama:** **TODOS** los diagramas deben ser generados en formato **Mermaid** y envueltos en sus respectivos bloques de código (ej: \`\`\`mermaid\n...\n\`\`\`).
3.  **Contenido a Generar e Insertar:**

    * ### 1. 🚶 Diagram Walkthrough (Diagrama de Flujo del Proceso Principal)
        * **Objetivo:** Mostrar una **vista de alto nivel** del flujo de ejecución principal del sistema (ej: "Usuario inicia Request" -> "API Gateway" -> "Servicio A" -> "Base de Datos").
        * **Formato:** Diagrama de Flujo (`flowchart` o `graph`).

    * ### 2. 🗺️ System Workflow (Flujo de Trabajo Detallado / Secuencia de Eventos)
        * **Objetivo:** Mostrar la **interacción detallada y la secuencia** de llamadas o eventos entre los componentes clave durante una operación crítica.
        * **Formato:** Diagrama de Secuencia (`sequenceDiagram`).

    * ### 3. 🏗️ Architecture Components (Componentes de Arquitectura)
        * **Objetivo:** Ilustrar la **estructura estática y las dependencias** de los principales módulos, servicios o capas (Frontend, Backend, DB, Microservicio X, Librería Y).
        * **Formato:** Diagrama de Bloques / Componentes (`C4 Diagram`, si es posible, o `mindmap` / `graph` de componentes).

    * ### 4. ⚙️ Container Lifecycle (Ciclo de Vida)
        * **Objetivo:** Describir en dos subsecciones (usando listas o texto, *no* necesariamente un diagrama de Mermaid, a menos que un `flowchart` simple sea apropiado) el ciclo de vida de la aplicación.
            * **Subsección a:** **Build Process** (Pasos clave para construir la imagen/artefacto, ej: `Dockerfile` steps).
            * **Subsección b:** **Runtime Process** (Pasos que ocurren desde que el contenedor/proceso se inicia hasta que está listo, ej: Inicialización de DB, Carga de Configuración).

    * ### 5. 📂 File-by-File Guide (Guía Archivo por Archivo)
        * **Objetivo:** Crear una tabla o una lista organizada que enumere los archivos y carpetas clave del proyecto, proporcionando una **descripción de una sola frase** sobre su propósito y contenido para una navegación rápida.

**Output Requerido:**

Proporciona únicamente el **contenido FINAL y completo del archivo README.md** actualizado. Asegúrate de que los bloques de código Mermaid sean válidos.
