#!/bin/bash

set -e  # Exit on error

echo "================================"
echo "Complete MLflow Restart"
echo "================================"
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if MLflow is responding
check_mlflow() {
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for MLflow to respond..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f http://localhost:5000/api/2.0/mlflow/experiments/list >/dev/null 2>&1; then
            echo -e "${GREEN}✓ MLflow is responding!${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ MLflow did not respond after $max_attempts attempts${NC}"
    return 1
}

# Stop everything
echo "1. Stopping services..."
docker compose down
echo -e "${GREEN}✓ Stopped${NC}"
echo ""

# Remove the container if it exists
echo "2. Removing old container..."
docker rm -f mlflow_server 2>/dev/null || true
echo -e "${GREEN}✓ Container removed${NC}"
echo ""

# Create local directories
echo "3. Creating local directories..."
mkdir -p mlflow_data mlflow_artifacts
chmod -R 777 mlflow_data mlflow_artifacts
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Build image
echo "4. Building MLflow image (this may take a minute)..."
docker compose build --no-cache mlflow
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo ""

# Start MLflow
echo "5. Starting MLflow container..."
docker compose up -d mlflow
echo -e "${GREEN}✓ Container started${NC}"
echo ""

# Wait a moment
echo "6. Waiting for initialization..."
sleep 5
echo ""

# Show logs
echo "7. Recent logs:"
echo "---"
docker compose logs --tail=30 mlflow
echo "---"
echo ""

# Check if container is running
if docker ps | grep -q mlflow_server; then
    echo -e "${GREEN}✓ Container is running${NC}"
else
    echo -e "${RED}✗ Container is not running${NC}"
    echo "Full logs:"
    docker compose logs mlflow
    exit 1
fi
echo ""

# Test connection
echo "8. Testing MLflow connection..."
if check_mlflow; then
    echo ""
    echo "================================"
    echo -e "${GREEN}SUCCESS!${NC}"
    echo "================================"
    echo ""
    echo "MLflow UI: http://localhost:5000"
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:5000/api/2.0/mlflow/experiments/list"
    echo ""
else
    echo ""
    echo "================================"
    echo -e "${RED}FAILED${NC}"
    echo "================================"
    echo ""
    echo "MLflow is not responding. Debugging info:"
    echo ""
    
    echo "Container status:"
    docker ps -a | grep mlflow
    echo ""
    
    echo "Full logs:"
    docker-compose logs mlflow
    echo ""
    
    echo "Try running manually:"
    echo "  docker-compose exec mlflow mlflow server --help"
    exit 1
fi
