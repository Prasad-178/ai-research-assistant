variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "The name of the project, used for naming resources."
  type        = string
  default     = "ai-research-assistant"
}

variable "ecr_image_uri" {
  description = "The full URI of the Docker image in ECR including the tag (e.g., public.ecr.aws/alias/repo:latest)."
  type        = string
  # This must be provided by the user or via -var option, as it's unique to your ECR repo.
  # Example: "public.ecr.aws/x5h9x8z0/prasadjs178/mlops-project:latest"
}

variable "model_s3_bucket" {
  description = "The name of the S3 bucket where the model is stored."
  type        = string
}

variable "model_s3_key_prefix" {
  description = "The S3 key prefix (path) to the model files within the bucket."
  type        = string
}

variable "ecs_task_cpu" {
  description = "The amount of CPU to allocate to the ECS task (in CPU units)."
  type        = number
  default     = 1024 # 1 vCPU
}

variable "ecs_task_memory" {
  description = "The amount of memory to allocate to the ECS task (in MiB)."
  type        = number
  default     = 8192 # 8GB
} 