variable "ssh_password" {
  description = "SSH root password"
  type        = string
  default     = "Ch@ng3M3N0w!2024"
  sensitive   = true
}

variable "portainer_password" {
  description = "Portainer admin password"
  type        = string
  default     = "Adm1nP@ssw0rd!"
  sensitive   = true
}

variable "grafana_password" {
  description = "Grafana admin password"
  type        = string
  default     = "Gr@f@n@Adm1n!"
  sensitive   = true
}

variable "domain_name" {
  description = "Domain name for SSL certificates"
  type        = string
  default     = "localhost"
}

variable "email" {
  description = "Email for Let's Encrypt certificates"
  type        = string
  default     = "admin@example.com"
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 7
}

variable "monitoring_enabled" {
  description = "Enable monitoring stack"
  type        = bool
  default     = true
}

variable "ssl_enabled" {
  description = "Enable SSL/TLS"
  type        = bool
  default     = false
}