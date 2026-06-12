variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Vertex AI and Artifact Registry"
  type        = string
}

variable "results_bucket_name" {
  description = "GCS bucket name for storing screening results"
  type        = string
}

variable "artifact_registry_repo" {
  description = "Artifact Registry repository ID for Docker images"
  type        = string
  default     = "thesis-models"
}

variable "create_artifact_registry" {
  description = "Set to false to use an existing Artifact Registry repository instead of creating one"
  type        = bool
  default     = false
}
