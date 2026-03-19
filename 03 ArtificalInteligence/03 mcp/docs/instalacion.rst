Instalación
============

Prerrequisitos
--------------

Antes de instalar el proyecto, asegúrate de tener instalados:

- **Docker**: Versión 20.10 o superior
- **Docker Compose**: Versión 1.29 o superior
- **Python 3.8+** (opcional, para generar documentación)

Verificación de Prerrequisitos
------------------------------

Ejecuta el script de verificación incluido:

.. code-block:: bash

   make check

O manualmente:

.. code-block:: bash

   ./check-prerequisites.sh

Instalación del Proyecto
------------------------

1. Clona o descarga el proyecto
2. Navega al directorio del proyecto
3. Ejecuta la verificación de prerrequisitos
4. Inicia el inspector

Instalación de Dependencias para Documentación
-----------------------------------------------

Si deseas generar la documentación localmente:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt

Generación de Documentación
---------------------------

Para generar la documentación HTML:

.. code-block:: bash

   cd docs
   make html

La documentación se generará en ``docs/_build/html/index.html``