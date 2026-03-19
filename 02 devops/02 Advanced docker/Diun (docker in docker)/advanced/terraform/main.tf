terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Network
resource "docker_network" "dind_network" {
  name   = "dind-network"
  driver = "bridge"

  ipam_config {
    subnet = "172.20.0.0/16"
  }
}

# Volumes
resource "docker_volume" "dind_data" {
  name = "dind_dind-data"
}

resource "docker_volume" "portainer_data" {
  name = "portainer_data"
}

resource "docker_volume" "grafana_data" {
  name = "grafana_data"
}

resource "docker_volume" "prometheus_data" {
  name = "prometheus_data"
}

# DinD Container
resource "docker_image" "dind" {
  name = "docker_in_docker-dind:latest"
  build {
    context    = ".."
    dockerfile = "docker/Dockerfile"
    tag        = ["docker_in_docker-dind:latest"]
  }
}

resource "docker_container" "dind" {
  name  = "dind"
  image = docker_image.dind.image_id

  privileged = true

  ports {
    internal = 9000
    external = 9003
  }

  ports {
    internal = 50422
    external = 50422
  }

  ports {
    internal = 9100
    external = 9100
  }

  volumes {
    host_path      = "/var/run/docker.sock"
    container_path = "/var/run/docker.sock"
  }

  volumes {
    volume_name    = docker_volume.dind_data.name
    container_path = "/var/lib/docker"
  }

  env = [
    "DOCKER_TLS_CERTDIR=",
    "SSH_PASSWORD=${var.ssh_password}",
    "PORTAINER_ADMIN_PASSWORD=${var.portainer_password}"
  ]

  networks_advanced {
    name = docker_network.dind_network.name
  }

  healthcheck {
    test     = ["CMD", "docker", "ps"]
    interval = "30s"
    timeout  = "10s"
    retries  = 3
  }

  restart = "unless-stopped"
}

# Documentation Container
resource "docker_image" "docs" {
  name = "docker_in_docker-docs:latest"
  build {
    context    = "../docs"
    dockerfile = "Dockerfile"
    tag        = ["docker_in_docker-docs:latest"]
  }
}

resource "docker_container" "docs" {
  name    = "docs-server"
  image   = docker_image.docs.image_id
  depends_on = [docker_container.dind]

  ports {
    internal = 80
    external = 8082
  }

  networks_advanced {
    name = docker_network.dind_network.name
  }

  healthcheck {
    test     = ["CMD", "curl", "-f", "http://localhost/"]
    interval = "30s"
    timeout  = "10s"
    retries  = 3
  }

  restart = "unless-stopped"
}

# Prometheus Container
resource "docker_container" "prometheus" {
  name    = "prometheus"
  image   = "prom/prometheus:latest"
  depends_on = [docker_container.dind]

  ports {
    internal = 9090
    external = 9090
  }

  volumes {
    host_path      = "${path.module}/../config/prometheus.yml"
    container_path = "/etc/prometheus/prometheus.yml"
  }

  volumes {
    volume_name    = docker_volume.prometheus_data.name
    container_path = "/prometheus"
  }

  command = [
    "--config.file=/etc/prometheus/prometheus.yml",
    "--storage.tsdb.path=/prometheus",
    "--web.console.libraries=/etc/prometheus/console_libraries",
    "--web.console.templates=/etc/prometheus/consoles",
    "--storage.tsdb.retention.time=200h",
    "--web.enable-lifecycle"
  ]

  networks_advanced {
    name = docker_network.dind_network.name
  }

  restart = "unless-stopped"
}

# Grafana Container
resource "docker_container" "grafana" {
  name    = "grafana"
  image   = "grafana/grafana:latest"
  depends_on = [docker_container.prometheus]

  ports {
    internal = 3000
    external = 3000
  }

  env = [
    "GF_SECURITY_ADMIN_PASSWORD=${var.grafana_password}",
    "GF_USERS_ALLOW_SIGN_UP=false"
  ]

  volumes {
    volume_name    = docker_volume.grafana_data.name
    container_path = "/var/lib/grafana"
  }

  networks_advanced {
    name = docker_network.dind_network.name
  }

  restart = "unless-stopped"
}