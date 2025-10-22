output "dind_container_id" {
  description = "DinD container ID"
  value       = docker_container.dind.id
}

output "portainer_url" {
  description = "Portainer web interface URL"
  value       = "http://localhost:9003"
}

output "documentation_url" {
  description = "Documentation URL"
  value       = "http://localhost:8082"
}

output "prometheus_url" {
  description = "Prometheus URL"
  value       = "http://localhost:9090"
}

output "grafana_url" {
  description = "Grafana URL"
  value       = "http://localhost:3000"
}

output "ssh_connection" {
  description = "SSH connection details"
  value       = "ssh root@localhost -p 50422"
}

output "network_name" {
  description = "Docker network name"
  value       = docker_network.dind_network.name
}

output "volumes" {
  description = "Created Docker volumes"
  value = {
    dind_data      = docker_volume.dind_data.name
    portainer_data = docker_volume.portainer_data.name
    grafana_data   = docker_volume.grafana_data.name
    prometheus_data = docker_volume.prometheus_data.name
  }
}