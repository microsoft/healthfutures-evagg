resource "azurerm_resource_group" "rg" {
  name     = "${var.deployment_name}-rg"
  location = var.location
  lifecycle {
    ignore_changes = [tags]
  }
}

data "azuread_users" "authorized_users" {
  mail_nicknames = values(var.authorized_users)
}

output "authorized_users" {
  description = "List of authorized users."
  value       = [for user in data.azuread_users.authorized_users.users : user.mail]
}
