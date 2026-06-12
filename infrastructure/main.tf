resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

module "vertexai" {
  source = "./modules/vertexai"

  project_id          = var.project_id
  region              = var.region
  results_bucket_name = var.gcs_results_bucket
}

