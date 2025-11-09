#!/usr/bin/env python3
import subprocess
import pandas as pd

PASSWORD = "password"
BASE_IP = "192.168.1.137"
PORT = "50422"

def get_ip(worker_name):
    container_name = f"some_container-{worker_name.replace('worker', 'worker-')}"
    cmd = f"docker inspect {container_name} --format '{{{{.NetworkSettings.Networks.diun.IPAddress}}}}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    ip_result = result.stdout.strip()
    if ip_result:
        return ip_result
    else:
        return "No_Disponible"

def get_container_statuses(worker_name):
    container_name = f"some_container-{worker_name.replace('worker', 'worker-')}"
    
    # Check if Docker daemon is running
    cmd_info = f"docker exec {container_name} docker info"
    result_info = subprocess.run(cmd_info, shell=True, capture_output=True, text=True)
    if result_info.returncode != 0:
        return {"Docker_Status": "Not Running", "portainer": "N/A", "filebrowser": "N/A", "nginx80": "N/A", "nginx443": "N/A", "NVIDIA": "N/A"}
    
    statuses = {"Docker_Status": "Running"}
    
    required = ["portainer", "filebrowser", "nginx80", "nginx443", "NVIDIA"]
    for cont in required:
        cmd = f"docker exec {container_name} docker ps -a --filter name={cont} --format '{{{{.Status}}}}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            statuses[cont] = "Not Found"
        else:
            status = result.stdout.strip()
            if 'Up' in status:
                statuses[cont] = "Running"
            elif 'Created' in status:
                statuses[cont] = "Created"
            else:
                statuses[cont] = "Stopped"
    
    return statuses

def main():
    data = []
    for i in range(1, 51):
        worker_name = f"worker{i}"
        print(f"Processing {worker_name}...")
        ip = get_ip(worker_name)
        statuses = get_container_statuses(worker_name)
        row = {
            "Worker": worker_name,
            "IP_Address": ip,
            **statuses
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv("report.csv", index=False)
    print("Report saved to report.csv")

if __name__ == "__main__":
    main()