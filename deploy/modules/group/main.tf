resource "azuread_group" "group" {
  display_name     = var.name
  security_enabled = true
}

locals {
  IS_GUID = "[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}"
}

data "azuread_service_principals" "sps" {
  application_ids = [for sp in var.member_names : sp if can(regex(local.IS_GUID, sp))]
}

data "azuread_users" "users" {
  mail_nicknames = [for name in var.member_names : name if !can(regex(local.IS_GUID, name))]
}

# There seems to be a terraform bug causing an error here on apply ("Invalid for_each argument").
# Commenting this block out and running, then commenting back in and rerunning seems to work.
resource "azuread_group_member" "members" {
  for_each = toset(concat(
    data.azuread_service_principals.sps.object_ids,
    data.azuread_users.users.object_ids
  ))
  group_object_id  = azuread_group.group.id
  member_object_id = each.key
}
