#!/bin/bash

# Nombre del archivo de salida
OUTPUT_FILE="workers_ips.csv"

# Limpia el archivo si existe y añade la cabecera CSV
echo "Worker,IP_Address" > "$OUTPUT_FILE"

# Contraseña y base de IP (asegúrate de que esta IP es el punto de acceso para todos los workers)
PASSWORD="password"
BASE_IP="192.168.1.84"
PORT="50422"

echo "Iniciando escaneo... los errores de SSH (ej. 'Host key verification failed') se ignorarán."

# Bucle robusto (C-style) para iterar del worker1 al worker50
for ((i=1; i<=50; i++))
do
    WORKER_NAME="worker$i"
    
    # El comando de SSH se mantiene. Se agrega el uso de comillas para las variables de seguridad.
    IP_RESULT=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$WORKER_NAME@$BASE_IP" -p "$PORT" 'ifconfig eth0 | grep "inet addr" | awk "{print \$2}" | cut -d: -f2' 2>/dev/null | head -n 2 | tail -n 1)

    # Verifica si se obtuvo un resultado (la IP)
    if [ -n "$IP_RESULT" ]; then
        # Elimina cualquier espacio en blanco extra y guarda la línea en el CSV
        CLEAN_IP=$(echo "$IP_RESULT" | tr -d '[:space:]')
        echo "$WORKER_NAME,$CLEAN_IP" >> "$OUTPUT_FILE"
        echo "✅ IP obtenida para $WORKER_NAME: $CLEAN_IP"
    else
        echo "❌ No se pudo conectar o no se encontró IP para $WORKER_NAME"
        echo "$WORKER_NAME,No_Disponible" >> "$OUTPUT_FILE"
    fi
done

echo ""
echo "---"
echo "✅ Proceso completado. La información se guardó en: $OUTPUT_FILE"

# Muestra el contenido final
cat "$OUTPUT_FILE"