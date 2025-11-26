#!/bin/bash

echo "================================"
echo "Fixing MLflow SQLite Issue"
echo "================================"
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Stop all services
echo "1. Stopping all services..."
docker compose down
echo -e "${GREEN}✓ Services stopped${NC}"
echo ""

# Remove volumes (this will delete existing MLflow data)
echo "2. Removing old volumes..."
read -p "This will delete existing MLflow data. Continue? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker volume rm callcenterai_mlflow_data 2>/dev/null || true
    docker volume rm callcenterai_mlflow_artifacts 2>/dev/null || true
    echo -e "${GREEN}✓ Volumes removed${NC}"
else
    echo -e "${YELLOW}Skipping volume removal${NC}"
fi
echo ""

# Rebuild MLflow image
echo "3. Rebuilding MLflow image..."
docker compose build --no-cache mlflow
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MLflow image rebuilt${NC}"
else
    echo "Failed to rebuild image"
    exit 1
fi
echo ""

# Start MLflow
echo "4. Starting MLflow..."
docker compose up -d mlflow
echo -e "${GREEN}✓ MLflow started${NC}"
echo ""

# Wait for MLflow to initialize
echo "5. Waiting for MLflow to initialize (30 seconds)..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Check status
echo "6. Checking MLflow status..."
docker compose logs --tail=10 mlflow
echo ""

# Test connection
echo "7. Testing connection..."
sleep 2
if curl -s http://localhost:5000 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ MLflow is accessible at http://localhost:5000${NC}"
else
    echo "MLflow is not responding yet. Check logs with:"
    echo "  docker-compose logs -f mlflow"
fi
echo ""

echo "================================"
echo "Done!"
echo "================================"
echo ""
echo "Access MLflow UI: http://localhost:5000"
echo "View logs: docker-compose logs -f mlflow"
echo ""
