# TODO: hardcoded for now, should be parameterized and turned into a managed resource.
data "azurerm_cognitive_account" "aoai_service" {
  resource_group_name = "aoai-dev"
  name                = "bmcaoai2"
}

resource "azurerm_role_assignment" "aoai_service" {
  for_each             = toset([for user in data.azuread_users.authorized_users.users : user.object_id])
  scope                = data.azurerm_cognitive_account.aoai_service.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = each.value
}

output "AZURE_OPENAI_ENDPOINT" {
  description = "The endpoint for the AOAI chat service."
  value       = data.azurerm_cognitive_account.aoai_service.endpoint
}