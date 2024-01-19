variable "deployment_name" {
  description = "Master deployment name, used as a prefix to derive various resource names."
  type        = string
  validation {
    condition = alltrue([
      can(regex("^[0-9a-z]+$", var.deployment_name)),
      length(var.deployment_name) >= 8,
      length(var.deployment_name) <= 16
    ])
    error_message = "Variable deployment_name must be 8-16 characters lowercase alphanumeric."
  }
}

variable "location" {
  description = "Azure region in which to create all resources."
  type        = string
}

variable "tenant_id" {
  description = "Tenant in which to create all resources."
  type        = string
}

variable "subscription_id" {
  description = "Subscription in which to create all resources."
  type        = string
}
