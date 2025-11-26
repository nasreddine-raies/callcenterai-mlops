#!/bin/bash

echo "================================"
echo "MLflow Setup Script"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and docker-compose are installed${NC}"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p notebooks
mkdir -p monitoring
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Stop any running containers
echo "Stopping existing containers..."
docker compose down
echo -e "${GREEN}✓ Containers stopped${NC}"
echo ""

# Build and start services
echo "Building Docker images..."
docker compose build mlflow

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MLflow image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build MLflow image${NC}"
    exit 1
fi

echo ""
echo "Building TF-IDF service..."
docker compose build tfidf_svc

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TF-IDF service image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build TF-IDF service image${NC}"
    exit 1
fi

echo ""
echo "Starting services..."
docker compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 5

# Check if MLflow is running
echo ""
echo "Checking MLflow service..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MLflow is running at http://localhost:5000${NC}"
else
    echo -e "${YELLOW}⚠ MLflow may still be starting up. Check logs with: docker-compose logs mlflow${NC}"
fi

# Check if TF-IDF service is running
echo ""
echo "Checking TF-IDF service..."
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ TF-IDF service is running at http://localhost:8001${NC}"
else
    echo -e "${YELLOW}⚠ TF-IDF service may still be starting up. Check logs with: docker-compose logs tfidf_svc${NC}"
fi

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Open MLflow UI: http://localhost:5000"
echo "2. Check API docs: http://localhost:8001/docs"
echo "3. Train your model: python src/tfidf_svc/train.py"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo ""
