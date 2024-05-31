output "EVAGG_CONTENT_CACHE_ENDPOINT" {
  description = "The endpoint for the CosmosDB cache."
  value       = azurerm_cosmosdb_account.db.endpoint
}

output "EVAGG_CONTENT_CACHE_CREDENTIAL" {
  description = "The primary key for the CosmosDB cache."
  value       = azurerm_cosmosdb_account.db.primary_key
  sensitive   = true
}

output "authorized_users" {
  description = "List of authorized users."
  value       = [for user in data.azuread_users.users.users : user.mail]
}
