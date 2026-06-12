resource "google_project_service" "aiplatform" {
  project            = var.project_id
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "models" {
  count         = var.create_artifact_registry ? 1 : 0
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry]
}

data "google_artifact_registry_repository" "existing" {
  count         = var.create_artifact_registry ? 0 : 1
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo
}

resource "google_storage_bucket" "results" {
  project       = var.project_id
  name          = var.results_bucket_name
  location      = var.region
  force_destroy = false

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}
