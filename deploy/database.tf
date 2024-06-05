resource "azurerm_cosmosdb_account" "db" {
  name                = "${var.deployment_name}-db"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  enable_automatic_failover     = true
  local_authentication_disabled = true

  capabilities {
    name = "EnableServerless"
  }

  consistency_policy {
    consistency_level       = "BoundedStaleness"
    max_interval_in_seconds = 10
    max_staleness_prefix    = 200
  }

  geo_location {
    location          = azurerm_resource_group.rg.location
    failover_priority = 0
  }
}

resource "azurerm_cosmosdb_sql_database" "db" {
  name                = "document_cache"
  resource_group_name = azurerm_resource_group.rg.name
  account_name        = azurerm_cosmosdb_account.db.name
}

resource "azurerm_cosmosdb_sql_container" "cache" {
  name                  = "cache"
  resource_group_name   = azurerm_resource_group.rg.name
  account_name          = azurerm_cosmosdb_account.db.name
  database_name         = azurerm_cosmosdb_sql_database.db.name
  partition_key_path    = "/id"
  partition_key_version = 1
}

resource "azurerm_cosmosdb_sql_container" "secondary_cache" {
  name                  = "secondary_cache"
  resource_group_name   = azurerm_resource_group.rg.name
  account_name          = azurerm_cosmosdb_account.db.name
  database_name         = azurerm_cosmosdb_sql_database.db.name
  partition_key_path    = "/id"
  partition_key_version = 1
}

# Create custom role limited to cache data read/write operations.
resource "azurerm_cosmosdb_sql_role_definition" "cache_role" {
  name                = "cache_role"
  resource_group_name = azurerm_resource_group.rg.name
  account_name        = azurerm_cosmosdb_account.db.name
  type                = "CustomRole"
  assignable_scopes   = [azurerm_cosmosdb_account.db.id]

  permissions {
    data_actions = [
      "Microsoft.DocumentDB/databaseAccounts/readMetadata",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/executeQuery",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/read",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/create",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/replace",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/upsert",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/delete"
    ]
  }
}

# Assign the custom role to all authorized users.
resource "azurerm_cosmosdb_sql_role_assignment" "cache_role_assignment" {
  for_each            = toset([for user in data.azuread_users.authorized_users.users : user.object_id])
  resource_group_name = azurerm_resource_group.rg.name
  account_name        = azurerm_cosmosdb_account.db.name
  role_definition_id  = azurerm_cosmosdb_sql_role_definition.cache_role.id
  scope               = azurerm_cosmosdb_account.db.id
  principal_id        = each.value
}

output "EVAGG_CONTENT_CACHE_ENDPOINT" {
  description = "The endpoint for the CosmosDB cache."
  value       = azurerm_cosmosdb_account.db.endpoint
}
