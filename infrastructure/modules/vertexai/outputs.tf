output "results_bucket_name" {
  description = "GCS bucket name where screening results are stored"
  value       = google_storage_bucket.results.name
}

output "artifact_registry_host" {
  description = "Artifact Registry hostname for Docker push/pull"
  value       = "${var.region}-docker.pkg.dev"
}

output "image_uri_prefix" {
  description = "Full image URI prefix for pushing Docker images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo}"
}
