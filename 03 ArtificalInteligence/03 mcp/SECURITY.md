# Política de Seguridad

## Reportar Vulnerabilidades

Si encuentras una vulnerabilidad de seguridad en este proyecto, por favor repórtala de manera responsable.

**No publiques vulnerabilidades en issues públicos.**

### Cómo Reportar

1. Envía un email a [tu-email@ejemplo.com] con detalles de la vulnerabilidad
2. Incluye:
   - Descripción clara del problema
   - Pasos para reproducir
   - Impacto potencial
   - Sugerencias de solución (opcional)

### Proceso de Respuesta

- Confirmaremos recepción dentro de 48 horas
- Investigaremos y te mantendremos informado del progreso
- Publicaremos un fix tan pronto como sea posible
- Te daremos crédito por el reporte (con tu permiso)

## Consideraciones de Seguridad

### Docker

- Este proyecto ejecuta contenedores Docker que pueden exponer procesos locales
- Por defecto, se enlaza solo a localhost
- No expongas los puertos a redes no confiables

### MCP Inspector

- El inspector incluye autenticación por defecto
- No deshabilites la autenticación en entornos de producción
- Mantén las imágenes actualizadas

### Configuraciones

- No commits configuraciones sensibles (.env.local)
- Usa variables de entorno para secrets
- Limita permisos de archivos de configuración

## Actualizaciones de Seguridad

Las actualizaciones de seguridad se publicarán en:
- Este documento
- El CHANGELOG.md
- Releases de GitHub

## Contacto

Para preguntas sobre seguridad: [tu-email@ejemplo.com]