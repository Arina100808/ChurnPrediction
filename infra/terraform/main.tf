resource "aws_s3_bucket" "model_bucket" {
  bucket = var.bucket_name

  tags = {
    Name        = var.bucket_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_s3_bucket_ownership_controls" "model_bucket_ownership" {
  bucket = aws_s3_bucket.model_bucket.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

