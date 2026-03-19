Convierte el documento de Word a formato Markdown manteniendo toda la estructura original. Sigue estos pasos:

1. VERIFICACIÓN INICIAL:
   - Lista los archivos en el directorio actual para confirmar el documento existe
   - Verifica si pandoc está instalado, si no, instálalo con: apt update && apt install -y pandoc

2. CONVERSIÓN PRINCIPAL:
   - Ejecuta: pandoc "[nombre-del-archivo.docx]" -t markdown -o "[nombre-del-archivo.md]" --extract-media=./images --wrap=preserve --standalone

3. VERIFICACIÓN POST-CONVERSIÓN:
   - Lista los archivos para confirmar se creó el .md y la carpeta images/
   - Muestra las primeras 50 líneas del archivo Markdown para verificar estructura
   - Confirma que las imágenes se extrajeron correctamente

4. REQUISITOS DE CALIDAD:
   - Mantener encabezados, tablas, listas y formato de texto
   - Preservar tabla de contenidos y enlaces internos
   - Extraer todas las imágenes con rutas correctas
   - Conservar metadatos del documento

Reporta el estado final de la conversión con cualquier error encontrado.

VARIACIONES ÚTILES:
- Para GitHub Flavored Markdown: usa -t gfm en lugar de -t markdown
- Para PDF: cambia la extensión a .pdf en el comando
- Para mejor preservación de formato: añade --reference-links
- Para documentos complejos: añade --toc para generar tabla de contenidos automática