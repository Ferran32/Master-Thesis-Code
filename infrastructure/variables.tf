variable "project_id" {
  description = "GCP project ID"
  type        = string

  validation {
    condition     = var.project_id != "your-gcp-project-id"
    error_message = "Set a real GCP project ID in terraform.tfvars."
  }
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "zone" {
  description = "GCP zone"
  type        = string
}

variable "gcs_results_bucket" {
  description = "GCS bucket name for Vertex AI screening results (must be globally unique)"
  type        = string
  default     = "thesis-screening-results"
}
