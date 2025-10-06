#!/bin/bash

# Kiểm tra Docker service
if ! service docker status > /dev/null; then
    echo "Starting Docker service..."
    sudo service docker start
fi

# Build và start services
docker-compose build
docker-compose up -d

# Kiểm tra trạng thái các services
echo "Checking services status..."
docker-compose ps

# Verify endpoints
echo "Verifying endpoints..."
curl -s http://localhost:6333/health # Qdrant
curl -s http://localhost:9000/minio/health # MinIO
curl -s http://localhost:5000 # MLflow