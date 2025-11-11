# test

el proyecto es simple:
    - descargo con docker pull una imagenes base que luego compartire con los contenedores worker
    - hago make up SCALE=n, donde n puede ser un numero entre 1 y 50
    - cada worker es un DinD (contenedores que tienen su propio docker daemon para ejecutar contenedores) y se le comparte la GPU, para que de ser necesario, estos contenedores puedan usar la GPU,
    - en cada worker que se inicia, corre el demonio de docker, con este demonio se inician 5 contenedores base: portainer-ce, filebrowser, nginx80, nginx443 y NVIDIA ( que ve la GPU del host)
    - luego de levantar los worker y esperar unos 60 segundos a que terminen de iniciarse se ejecuta un script de python que genera un reporte csv, en ese reporte, el script entra por ssh a cada worker a traer informacion que luego pondra en un csv, cada worker debe tener una ip y todos los servicios estar en "running", incluyendo el NVIDIA

# validar
    - hacer make up SCALE=4
    - que los worker se inicien con el docker daemon funcionando dentro de cada worker
    - que en cada worker sus contenedores esten ejecutandose
    - que en cada worker pueda hacer nvidia-smi
    - que en cada worker el contenedor NVIDIA vea la GPU
    - todas estas validaciones conectandose con ssh, ejemplo:
    sshpass -p 'password' ssh worker1@192.168.1.137 -p 50422 o sshpass -p 'password' ssh worker2@192.168.1.137 -p 50422 