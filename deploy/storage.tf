resource "azurerm_storage_account" "storage" {
  name                     = "${var.deployment_name}sa"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  enable_https_traffic_only       = true
  shared_access_key_enabled       = false
  public_network_access_enabled   = false
  allow_nested_items_to_be_public = false
}

resource "azurerm_role_assignment" "storage_roles" {
  for_each             = toset([for user in data.azuread_users.authorized_users.users : user.object_id])
  scope                = azurerm_storage_account.storage.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = each.value
}

output "AZURE_STORAGE_ACCOUNT_NAME" {
  description = "The name of the storage account."
  value       = azurerm_storage_account.storage.name
}
