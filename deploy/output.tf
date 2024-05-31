output "authorized_users" {
  description = "List of authorized users."
  value       = [for user in data.azuread_users.users.users : user.mail]
}
