terraform {
  backend "s3" {
    bucket         = "ai-research-assistant-tfstate"
    key            = "global/s3/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "ai-research-assistant-tf-lock-table"
    encrypt        = true
  }
} 