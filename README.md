# imagenesDocker

# Nota
- Tener en cuenta, si se desea que la hora dentro del docker sea la misma que la hora local del servidor, se puede seguir la guia en: https://diarioinforme.com/como-administrar-las-zonas-horarias-en-los-contenedores-de-docker/

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
 
