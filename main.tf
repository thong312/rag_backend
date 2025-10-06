provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "rag_server" {
  ami           = var.ami_id
  instance_type = "m7i-flex.xlarge"  # Tăng size để chạy nhiều model

  user_data = <<-EOF
              #!/bin/bash
              # Install Docker
              apt-get update
              apt-get install -y docker.io docker-compose
              systemctl start docker
              systemctl enable docker

              # Install Ollama
              curl https://ollama.ai/install.sh | sh
              systemctl start ollama

              # Pull models
              ollama pull qwen:3b
              ollama pull nomic-embed-text  # Thêm model embeddings

              # Mount EBS volume 
              mkfs -t ext4 /dev/xvdf
              mkdir -p /mnt/models
              mount /dev/xvdf /mnt/models
              echo "/dev/xvdf /mnt/models ext4 defaults 0 0" >> /etc/fstab

              # Update model path
              echo 'OLLAMA_MODELS=/mnt/models' >> /etc/environment
              systemctl restart ollama

              # Clone & Deploy app
              git clone ${var.repo_url}
              cd rag_project
              docker-compose up -d
              EOF

  root_block_device {
    volume_size = 30  # Tăng dung lượng ổ root
  }

  tags = {
    Name = "rag-server"
  }

  vpc_security_group_ids = [aws_security_group.rag_sg.id]
}

resource "aws_security_group" "rag_sg" {
  name = "rag-security-group"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8080
    to_port     = 8080 
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 6333
    to_port     = 6334
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 9000
    to_port     = 9001
    protocol    = "tcp" 
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ebs_volume" "model_storage" {
  availability_zone = aws_instance.rag_server.availability_zone
  size             = 40
  type             = "gp3"
  iops             = 3000
  throughput       = 125
  
  tags = {
    Name = "ollama-models"
  }
}

resource "aws_volume_attachment" "model_attach" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.model_storage.id
  instance_id = aws_instance.rag_server.id
}
