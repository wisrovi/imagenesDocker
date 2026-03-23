Para transformar tu requerimiento en un Prompt de Ingeniería de Calidad que OpenCode pueda ejecutar con precisión quirúrgica, he diseñado una estructura que define el comportamiento del script, la lógica de los mensajes y las restricciones de Git.

Aquí tienes el prompt optimizado:
Prompt: Generación de Script de Mapeo Granular para Git

Rol: Actúa como un Senior Release Engineer & Automation Expert. Tu tarea es crear un script robusto en Bash diseñado para realizar commits individuales y granulares por cada archivo del proyecto.
OBJETIVO DEL SCRIPT (mapper_files.sh):

El script debe recorrer de forma recursiva todo el directorio actual y mapear cada archivo individualmente a su propio commit de Git.
REGLAS DE EJECUCIÓN TÉCNICA:

    Granularidad Absoluta: Está estrictamente prohibido ejecutar git add <folder>. Cada archivo debe ser agregado individualmente mediante su ruta completa (git add path/to/file.py).

    Identificación de Cambios: El script debe detectar si el archivo es nuevo (untracked) o modificado para asignar el prefijo correcto.

    Lógica de Prefijos (Conventional Commits):

        [FEATURE]: Para archivos nuevos detectados.

        [REFACTOR]: Para cambios en la lógica de archivos existentes.

        [DOC]: Para archivos Markdown, textos o documentación.

        [FIX]: Si el contenido del archivo sugiere una corrección de errores.

        [CONFIG]: Para archivos de configuración (.yaml, .toml, .pylintrc, Makefile).

        [TEST]: Para archivos dentro de carpetas de test o con prefijo test_.

REQUISITOS DEL MENSAJE DE COMMIT:

    Idioma: Inglés técnico perfecto.

    Contenido: El mensaje debe explicar brevemente la función del archivo o el cambio realizado basándose en su nombre y extensión.

    Formato: [PREFIX] Action description for filename: technical detail.

TAREA PARA EL AGENTE:

    Generar el script mapper_files.sh que implemente una función recursiva o use find para procesar solo archivos (type f), excluyendo la carpeta .git y el propio script.

    Validación de Entorno: El script debe verificar si existe un repositorio git inicializado antes de proceder.

    No borrar nada: El script solo debe realizar acciones de add y commit.

    Historial: Al finalizar, el script debe imprimir un resumen de cuántos archivos fueron mapeados exitosamente.
