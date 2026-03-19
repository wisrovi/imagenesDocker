#!/usr/bin/env python3
import subprocess
import csv

# Nombre del archivo de salida
OUTPUT_FILE = "workers_ips.csv"

# Contraseña y base de IP
PASSWORD = "password"
BASE_IP = "192.168.1.84"
PORT = "50422"

print("Iniciando escaneo... los errores de SSH (ej. 'Host key verification failed') se ignorarán.")

# Abre el archivo CSV y escribe la cabecera
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Worker", "IP_Address"])

    # Bucle para iterar del worker1 al worker50
    for i in range(1, 51):
        worker_name = f"worker{i}"
        
        # Comando SSH
        cmd = f"sshpass -p {PASSWORD} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {worker_name}@{BASE_IP} -p {PORT} 'ifconfig eth0 | grep \"inet addr\" | awk \"{{print \\$2}}\" | cut -d: -f2' 2>/dev/null | head -n 2 | tail -n 1"
        
        # Ejecuta el comando
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ip_result = result.stdout.strip()
        
        # Verifica si se obtuvo un resultado
        if ip_result:
            # Elimina espacios en blanco extra
            clean_ip = ip_result.replace(' ', '')
            writer.writerow([worker_name, clean_ip])
            print(f"✅ IP obtenida para {worker_name}: {clean_ip}")
        else:
            writer.writerow([worker_name, "No_Disponible"])
            print(f"❌ No se pudo conectar o no se encontró IP para {worker_name}")

print("")
print("---")
print(f"✅ Proceso completado. La información se guardó en: {OUTPUT_FILE}")

# Muestra el contenido final
# with open(OUTPUT_FILE, 'r') as f:
#     print(f.read())