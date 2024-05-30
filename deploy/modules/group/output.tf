output "object_id" {
  value = azuread_group.group.object_id
}

output "login_ids" {
  value = concat(
    data.azuread_service_principals.sps.application_ids,
    [for user in data.azuread_users.users.users : user.mail]
  )
}
