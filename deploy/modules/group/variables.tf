variable "name" {
  type = string
}

variable "member_names" {
  description = "List of members of this group - can be a mix of user email names and service principal AppIDs."
  type        = list(string)
}
