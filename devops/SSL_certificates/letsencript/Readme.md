modificar el archivo: create_dns_01.sh, cambiando los datos por los de interes

Explicación:
certonly: Solo genera el certificado, no instala nada.
--webroot-path: Usa el método HTTP-01 (necesita que tu Nginx sirva un directorio accesible por /.well-known/acme-challenge/).
--email: Correo de contacto para renovación o problemas.
--agree-tos: Acepta automáticamente los términos de Let's Encrypt.
-d: Dominio para el que se genera el certificado.
Los certificados se guardan en /etc/letsencrypt, mapeados a ./nginx/certs/....



Luego ejecutar el archivo: con `sudo sh create_dns_01.sh`

mientras se ejecuta, se detendra un momento esperando que se de enter, aca se debe detener y crear un txt en el dominio:
# Crear el registro TXT en Ionos

Entra al panel de administración de tu dominio en Ionos.

Ve a la sección de DNS o Zona DNS para <subdominio o dominio a crear certificados>.

Crea un nuevo registro TXT con:

Nombre/Host: _acme-challenge.<subdominio o dominio a crear certificados>
Valor: <valor dado por durante la ejecucion del script>

TTL: puedes dejar el valor por defecto o 3600 segundos.

Nota: Algunos paneles permiten solo _acme-challenge si el registro está dentro de la zona www.security.ecapturedtech.com. Asegúrate de que quede exactamente como Certbot indica.




La propagación puede tardar desde unos segundos hasta varios minutos, para confirmar si ya esta visible el txt se puede usar:
https://toolbox.googleapps.com/apps/dig/#TXT/_acme-challenge.www.security.ecapturedtech.com

Una vez que el registro TXT esté visible, vuelve al contenedor de Certbot y presiona Enter.

NOTA: Certbot verificará el TXT y, si todo está correcto, generará los certificados en: `./nginx/certs/letsencrypt/live`