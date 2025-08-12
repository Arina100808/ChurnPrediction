variable "aws_region" {
  type        = string
  description = "AWS region to deploy resources"
  default     = "eu-central-1"
}

variable "bucket_name" {
  type        = string
  description = "Name of the S3 bucket where model artifacts will be stored"
  default     = "default-churn-model-artifacts"
}

variable "environment" {
  type        = string
  description = "Deployment environment (e.g. dev, test, prod)"
  default     = "dev"
}
