# imagenesDocker

# Nota

- Tener en cuenta, si se desea que la hora dentro del docker sea la misma que la hora local del servidor, se puede
  seguir la guia en: https://diarioinforme.com/como-administrar-las-zonas-horarias-en-los-contenedores-de-docker/

## en modo resumen:

### En el Dockerfile:

 ``` 
  FROM ...
  .
  .
  .
  ENV TZ=America/Bogota
  ENV DEBIAN_FRONTEND=noninteractive
  RUN apt-get install tzdata -y
  .
  .
  .
  CMD ...
  ```

### En el docker-compose:

  ```
  .
  .
  .
  volumes:
    - /etc/timezone:/etc/timezone:ro
    - /etc/localtime:/etc/localtime:ro
  .
  .
  .
  ```

# Arbol

- Portainer: devops/Code QA/Check _services_status/portained/
- jenkins: devops/Pipelines/Jenkins/
## Heavy files ignored

The repository now excludes large files that are not needed for version control. The following patterns have been added to **.gitignore**:

```
# Heavy files
*.zip
*.tar
*.sql
*.ibd
*.frm
*.data
```

These patterns prevent files such as Docker images, database dumps, and other large binaries from being tracked. Existing heavy files have been removed from the repository history.
