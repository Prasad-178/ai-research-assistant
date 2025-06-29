output "app_url" {
  description = "The public URL of the deployed application."
  value       = "http://${aws_lb.ml_app_lb.dns_name}"
} 