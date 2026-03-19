#!/usr/bin/env python3
"""
Kind Cluster Manager

A Python script for managing Kind clusters with GPU support.
Provides automated setup, monitoring, and cleanup functionality.
"""

import subprocess
import yaml
import argparse
import sys
from pathlib import Path


class KindClusterManager:
    """Manager class for Kind cluster operations."""

    def __init__(self, config_path: str = "config/kind-config.yaml"):
        self.config_path = Path(config_path)
        self.cluster_name = "kind"

    def run_command(self, command: list, capture_output: bool = False) -> str:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result.stdout if capture_output else ""
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(command)}")
            print(f"Error: {e}")
            return ""

    def create_cluster(self) -> bool:
        """Create a Kind cluster with the specified configuration."""
        if not self.config_path.exists():
            print(f"Configuration file not found: {self.config_path}")
            return False

        print("Creating Kind cluster...")
        cmd = ["kind", "create", "cluster", "--config", str(self.config_path)]
        try:
            self.run_command(cmd)
            print("Cluster created successfully!")
            return True
        except Exception as e:
            print(f"Failed to create cluster: {e}")
            return False

    def delete_cluster(self) -> bool:
        """Delete the Kind cluster."""
        print("Deleting Kind cluster...")
        cmd = ["kind", "delete", "cluster"]
        try:
            self.run_command(cmd)
            print("Cluster deleted successfully!")
            return True
        except Exception as e:
            print(f"Failed to delete cluster: {e}")
            return False

    def get_cluster_status(self) -> dict:
        """Get the status of the cluster."""
        status = {}

        # Check if cluster exists
        cmd = ["kind", "get", "clusters"]
        output = self.run_command(cmd, capture_output=True)
        status["exists"] = self.cluster_name in output

        if status["exists"]:
            # Get nodes
            cmd = ["kubectl", "get", "nodes", "-o", "json"]
            output = self.run_command(cmd, capture_output=True)
            if output:
                nodes = yaml.safe_load(output)
                status["nodes"] = len(nodes.get("items", []))
                status["node_status"] = [
                    {"name": node["metadata"]["name"], "status": node["status"]["conditions"][-1]["status"]}
                    for node in nodes.get("items", [])
                ]

        return status

    def install_gpu_plugin(self) -> bool:
        """Install NVIDIA GPU device plugin."""
        print("Installing NVIDIA GPU device plugin...")
        url = "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml"
        cmd = ["kubectl", "apply", "-f", url]
        try:
            self.run_command(cmd)
            print("GPU plugin installed successfully!")
            return True
        except Exception as e:
            print(f"Failed to install GPU plugin: {e}")
            return False

    def install_argocd(self) -> bool:
        """Install ArgoCD in the cluster."""
        print("Installing ArgoCD...")
        cmds = [
            ["kubectl", "create", "namespace", "argocd"],
            ["kubectl", "apply", "-n", "argocd", "-f",
             "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"]
        ]

        for cmd in cmds:
            try:
                self.run_command(cmd)
            except Exception as e:
                print(f"Failed to install ArgoCD: {e}")
                return False

        print("ArgoCD installed successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Kind Cluster Manager")
    parser.add_argument("action", choices=["create", "delete", "status", "install-gpu", "install-argocd"],
                       help="Action to perform")
    parser.add_argument("--config", default="config/kind-config.yaml",
                       help="Path to Kind configuration file")

    args = parser.parse_args()

    manager = KindClusterManager(args.config)

    if args.action == "create":
        success = manager.create_cluster()
        if success:
            manager.install_gpu_plugin()
            manager.install_argocd()
    elif args.action == "delete":
        manager.delete_cluster()
    elif args.action == "status":
        status = manager.get_cluster_status()
        print(yaml.dump(status, default_flow_style=False))
    elif args.action == "install-gpu":
        manager.install_gpu_plugin()
    elif args.action == "install-argocd":
        manager.install_argocd()


if __name__ == "__main__":
    main()