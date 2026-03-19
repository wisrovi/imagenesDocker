# Guía de Contribución

¡Gracias por tu interés en contribuir a este proyecto! Esta guía explica cómo puedes ayudar a mejorar el setup de MCP Inspector con Docker.

## Tipos de Contribuciones

- **Reportar bugs**: Usa los issues de GitHub para reportar problemas
- **Sugerir mejoras**: Comparte ideas para nuevas funcionalidades
- **Contribuir código**: Envía pull requests con mejoras
- **Mejorar documentación**: Actualiza o amplía la documentación

## Desarrollo Local

### Prerrequisitos

- Docker y Docker Compose instalados
- Python 3.8+ (para documentación)
- Git

### Configuración del Entorno

1. Clona el repositorio:
   ```bash
   git clone <url-del-repo>
   cd mcp
   ```

2. Verifica prerrequisitos:
   ```bash
   make check
   ```

3. Configura variables de entorno (opcional):
   ```bash
   cp .env .env.local
   # Edita .env.local según tus necesidades
   ```

### Estructura del Proyecto

```
mcp/
├── docker-compose.yml    # Configuración principal de Docker
├── .env                  # Variables de entorno
├── Makefile             # Comandos comunes
├── check-prerequisites.sh # Script de verificación
├── sample-config.json   # Ejemplo de configuración MCP
├── docs/                # Documentación Sphinx
├── README.md            # Documentación principal
├── LICENSE              # Licencia MIT
└── .gitignore          # Archivos ignorados por Git
```

## Flujo de Trabajo

### 1. Crear una Rama

```bash
git checkout -b feature/nueva-funcionalidad
# o
git checkout -b fix/problema-especifico
```

### 2. Hacer Cambios

- Sigue las convenciones de código existentes
- Actualiza la documentación si es necesario
- Añade tests si corresponde

### 3. Probar Cambios

```bash
make run
# Verifica que todo funcione correctamente
```

### 4. Commit y Push

```bash
git add .
git commit -m "Descripción clara del cambio"
git push origin feature/nueva-funcionalidad
```

### 5. Pull Request

- Crea un PR en GitHub
- Describe los cambios realizados
- Referencia issues relacionados

## Estándares de Código

### Docker

- Usa imágenes oficiales cuando sea posible
- Documenta todos los puertos expuestos
- Incluye healthchecks apropiados

### Documentación

- Escribe en español (idioma principal del proyecto)
- Usa formato RST para Sphinx
- Incluye ejemplos prácticos

### Commits

- Usa mensajes descriptivos en inglés
- Sigue el formato: `tipo: descripción breve`

Tipos comunes:
- `feat:` nueva funcionalidad
- `fix:` corrección de bug
- `docs:` cambios en documentación
- `style:` cambios de formato
- `refactor:` refactorización
- `test:` añadir tests
- `chore:` tareas de mantenimiento

## Reportar Issues

Cuando reportes un bug, incluye:

- **Descripción clara** del problema
- **Pasos para reproducir**
- **Comportamiento esperado** vs **comportamiento actual**
- **Entorno**: versión de Docker, OS, etc.
- **Logs** relevantes (usa `make logs`)

## Preguntas

Si tienes dudas, abre un issue con la etiqueta `question` o contacta al maintainer.

¡Gracias por contribuir! 🎉