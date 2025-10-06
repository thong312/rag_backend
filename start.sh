#!/bin/bash

# Start WSL services
sudo service docker start

# Initialize infrastructure with Terraform
terraform init
terraform apply -auto-approve

# Deploy với Ansible
ansible-playbook deploy.yml

echo "Environment is ready!"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
fi

# Start Docker service
sudo service docker start

# Check terraform
if ! command -v terraform &> /dev/null; then
    echo "Terraform not found. Please install terraform first."
    exit 1
fi

# Initialize infrastructure with Terraform
terraform init
terraform apply -auto-approve

# Wait for instance to be ready
echo "Waiting for instance to be ready..."
sleep 60

# Deploy with Ansible
ansible-playbook -i inventory.ini deploy.yml

echo "Environment is ready!"