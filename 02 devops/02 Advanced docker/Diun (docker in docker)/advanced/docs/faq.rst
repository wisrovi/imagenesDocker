Preguntas Frecuentes (FAQ)
==========================

.. _faq:

¿Es seguro usar Docker-in-Docker?
---------------------------------

DinD requiere modo privilegiado, lo que puede tener riesgos de seguridad. Úselo solo en entornos de desarrollo o testing controlados. Para producción, considere alternativas como Docker-outside-of-Docker (DooD).

¿Puedo usar esto en producción?
-------------------------------

No se recomienda para entornos de producción debido a los riesgos de seguridad y complejidad. Use contenedores normales o servicios gestionados como Kubernetes.

¿Cuál es la diferencia con Docker Compose normal?
-------------------------------------------------

Docker Compose ejecuta múltiples contenedores en el mismo host Docker. DinD ejecuta un daemon Docker completo dentro de un contenedor, permitiendo contenerización anidada.

¿Puedo cambiar los puertos expuestos?
-------------------------------------

Sí, edite el archivo ``docker-compose.yaml`` y cambie las asignaciones de puertos según sus necesidades.

¿Hay soporte para Windows?
--------------------------

Sí, funciona en Windows con WSL2 o Docker Desktop. Sin embargo, puede requerir configuraciones adicionales para modo privilegiado.

¿Cuántos recursos consume?
--------------------------

Aproximadamente 500MB-1GB de RAM y 5-10GB de disco por instancia. Escalee según sus necesidades.

¿Puedo usar volúmenes nombrados?
--------------------------------

Sí, puede modificar los volúmenes en ``docker-compose.yaml`` para usar volúmenes nombrados en lugar de bind mounts.

¿Hay logging integrado?
-----------------------

Sí, todos los servicios registran en stdout/stderr. Use ``docker-compose logs`` para ver los logs.

¿Puedo personalizar la configuración de SSH?
--------------------------------------------

Sí, modifique los scripts en ``scripts/`` para cambiar la configuración de SSH, como puerto, usuarios, etc.

¿Es compatible con Docker Swarm?
---------------------------------

El daemon interno puede unirse a un Swarm, pero requiere configuración adicional. No se recomienda para producción.

¿Hay soporte para GPU?
----------------------

No directamente. Para GPU, necesitaría pasar los dispositivos y configurar el contenedor apropiadamente.