#!/usr/bin/env python3
"""
GPU Resource Monitor for Kind Clusters

Monitors GPU usage and availability in Kubernetes pods.
Provides real-time statistics and alerts.
"""

import subprocess
import json
import time
from typing import Dict, List
import argparse


class GPUResourceMonitor:
    """Monitor GPU resources in a Kubernetes cluster."""

    def __init__(self):
        self.namespace = "default"

    def run_kubectl_command(self, command: List[str]) -> Dict:
        """Run a kubectl command and return parsed JSON output."""
        try:
            cmd = ["kubectl"] + command + ["-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed: kubectl {' '.join(command)}")
            print(f"Error: {e}")
            return {}
        except json.JSONDecodeError:
            print("Failed to parse JSON output")
            return {}

    def get_gpu_nodes(self) -> List[Dict]:
        """Get nodes with GPU resources."""
        nodes = self.run_kubectl_command(["get", "nodes"])
        gpu_nodes = []

        for node in nodes.get("items", []):
            capacity = node.get("status", {}).get("capacity", {})
            if "nvidia.com/gpu" in capacity:
                gpu_nodes.append({
                    "name": node["metadata"]["name"],
                    "gpu_capacity": int(capacity["nvidia.com/gpu"]),
                    "gpu_allocatable": int(node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0))
                })

        return gpu_nodes

    def get_gpu_pods(self) -> List[Dict]:
        """Get pods using GPU resources."""
        pods = self.run_kubectl_command(["get", "pods", "--all-namespaces"])
        gpu_pods = []

        for pod in pods.get("items", []):
            containers = pod.get("spec", {}).get("containers", [])
            for container in containers:
                requests = container.get("resources", {}).get("requests", {})
                limits = container.get("resources", {}).get("limits", {})

                gpu_request = requests.get("nvidia.com/gpu", "0")
                gpu_limit = limits.get("nvidia.com/gpu", "0")

                if gpu_request != "0" or gpu_limit != "0":
                    gpu_pods.append({
                        "namespace": pod["metadata"]["namespace"],
                        "name": pod["metadata"]["name"],
                        "node": pod.get("spec", {}).get("nodeName", "Unknown"),
                        "gpu_request": gpu_request,
                        "gpu_limit": gpu_limit,
                        "status": pod.get("status", {}).get("phase", "Unknown")
                    })
                    break  # Only count once per pod

        return gpu_pods

    def get_cluster_gpu_usage(self) -> Dict:
        """Get overall GPU usage statistics."""
        gpu_nodes = self.get_gpu_nodes()
        gpu_pods = self.get_gpu_pods()

        total_capacity = sum(node["gpu_capacity"] for node in gpu_nodes)
        total_allocatable = sum(node["gpu_allocatable"] for node in gpu_nodes)

        # Calculate used GPUs (simplified - assumes 1 GPU per pod for now)
        used_gpus = len([pod for pod in gpu_pods if pod["status"] == "Running"])

        return {
            "total_capacity": total_capacity,
            "total_allocatable": total_allocatable,
            "used_gpus": used_gpus,
            "available_gpus": total_allocatable - used_gpus,
            "utilization_percent": (used_gpus / total_allocatable * 100) if total_allocatable > 0 else 0,
            "nodes": gpu_nodes,
            "pods": gpu_pods
        }

    def print_status(self):
        """Print current GPU status."""
        usage = self.get_cluster_gpu_usage()

        print("=== GPU Resource Status ===")
        print(f"Total GPU Capacity: {usage['total_capacity']}")
        print(f"Total Allocatable GPUs: {usage['total_allocatable']}")
        print(f"Used GPUs: {usage['used_gpus']}")
        print(f"Available GPUs: {usage['available_gpus']}")
        print(".1f")

        print("\n=== GPU Nodes ===")
        for node in usage["nodes"]:
            print(f"- {node['name']}: {node['gpu_allocatable']}/{node['gpu_capacity']} GPUs available")

        print("\n=== GPU Pods ===")
        for pod in usage["pods"]:
            print(f"- {pod['namespace']}/{pod['name']} ({pod['status']}): {pod['gpu_limit']} GPUs")

    def monitor_continuous(self, interval: int = 30):
        """Continuously monitor GPU usage."""
        print(f"Starting continuous monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop")

        try:
            while True:
                self.print_status()
                print(f"\n--- Waiting {interval} seconds ---\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="GPU Resource Monitor for Kind Clusters")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Enable continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Monitoring interval in seconds (default: 30)")

    args = parser.parse_args()

    monitor = GPUResourceMonitor()

    if args.continuous:
        monitor.monitor_continuous(args.interval)
    else:
        monitor.print_status()


if __name__ == "__main__":
    main()