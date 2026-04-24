# 🌐 Remote Deployment Module

Este módulo está diseñado para desplegar y controlar el generador de tráfico en múltiples máquinas de una red local o remota a través de SSH.

## 📂 Contenido del Módulo

| Archivo | Propósito |
|---------|-----------|
| `pcs.conf` | Lista de direcciones IP de los equipos remotos. |
| `deploy.sh` | Script que automatiza la copia de archivos y la configuración inicial vía SCP/SSH. |
| `control.sh` | Permite ejecutar comandos maestros (start, stop, logs) de forma simultánea en todos los PCs remotos. |
| `Makefile` | Comandos unificados para gestionar el clúster de máquinas (`make deploy`, `make start`, `make status`). |

## 🔗 Prerrequisitos

Para que este módulo funcione correctamente, es necesario configurar el acceso SSH mediante claves públicas (`ssh-copy-id`) para evitar el ingreso manual de contraseñas durante el despliegue masivo.

## 🚀 Flujo de Trabajo Remoto

1. **Configurar Hosts**: Añadir las IPs de destino en `pcs.conf`.
2. **Despliegue**: `make deploy` enviará los archivos necesarios a cada host.
3. **Inicio Masivo**: `make start N=10` lanzará 10 réplicas en cada uno de los servidores remotos listados.
4. **Estado**: `make status` reportará cuántas instancias están operativas en cada nodo de la red.
