data "azurerm_cognitive_account" "aoai_service" {
  for_each = var.aoai_deployments
  resource_group_name = each.value
  name                = each.key
}

locals {
  # Make a list of all the authorized user/AOAI service pairs.
  role_assignments = flatten([
    for user in data.azuread_users.authorized_users.users : [
      for service in data.azurerm_cognitive_account.aoai_service : {
        user_id = user.object_id
        service_id = service.id
      }
    ]
  ])
}

# Add a role assignment for each authorized user for all the AOAI services.
resource "azurerm_role_assignment" "aoai_service" {
  for_each = {
    for assignment in local.role_assignments : "${assignment.user_id}-${assignment.service_id}" => assignment
  }
  scope                = each.value.service_id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = each.value.user_id
}

output "AZURE_OPENAI_ENDPOINTS" {
  description = "The endpoints for the AOAI chat services."
  value       = [for service in data.azurerm_cognitive_account.aoai_service : service.endpoint]
}
