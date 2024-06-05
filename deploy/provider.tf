terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=3.34.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "=2.31.0"
    }
  }
}

provider "azurerm" {
  features {}
  tenant_id           = var.tenant_id
  subscription_id     = var.subscription_id
  storage_use_azuread = true
}

provider "azuread" {
  tenant_id = var.tenant_id
}
