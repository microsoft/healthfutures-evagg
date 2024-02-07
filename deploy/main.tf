resource "azurerm_resource_group" "rg" {
  name     = "${var.deployment_name}-rg"
  location = var.location
  lifecycle {
    ignore_changes = [tags]
  }
}
