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
    cmd = f"docker exec {container_name} docker ps -a"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return {cont: "Error" for cont in ["portainer", "filebrowser", "nginx80", "nginx443", "NVIDIA"]}
    
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        return {cont: "No containers" for cont in ["portainer", "filebrowser", "nginx80", "nginx443", "NVIDIA"]}
    
    # Parse containers
    containers = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 7:
            name = parts[-1]
            status = ' '.join(parts[4:-1])  # STATUS column
            containers[name] = status
    
    required = ["portainer", "filebrowser", "nginx80", "nginx443", "NVIDIA"]
    statuses = {}
    for cont in required:
        if cont in containers:
            if 'Up' in containers[cont]:
                statuses[cont] = "Running"
            else:
                statuses[cont] = "Stopped"
        else:
            statuses[cont] = "Not Found"
    
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