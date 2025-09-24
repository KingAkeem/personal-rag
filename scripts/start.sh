#!/bin/bash

# Configuration
DEFAULT_GPU="auto"  # Change to "amd" or "nvidia" to force a specific type

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to detect GPU type
detect_gpu() {
    if lspci | grep -i nvidia > /dev/null; then
        echo "nvidia"
    elif lspci | grep -i "amd\|radeon" > /dev/null; then
        echo "amd"
    else
        echo "unknown"
    fi
}

# Function to check if NVIDIA Docker runtime is available
check_nvidia_runtime() {
    if docker info 2>/dev/null | grep -i nvidia > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Determine which compose file to use
if [ "$1" == "amd" ]; then
    COMPOSE_FILE="docker-compose.amd.yml"
    GPU_TYPE="amd"
    echo -e "${YELLOW}Using AMD configuration (forced)${NC}"
elif [ "$1" == "nvidia" ]; then
    COMPOSE_FILE="docker-compose.nvidia.yml"
    GPU_TYPE="nvidia"
    echo -e "${YELLOW}Using NVIDIA configuration (forced)${NC}"
else
    # Auto-detect
    DETECTED_GPU=$(detect_gpu)
    
    if [ "$DETECTED_GPU" == "nvidia" ]; then
        if check_nvidia_runtime; then
            COMPOSE_FILE="docker-compose.nvidia.yml"
            GPU_TYPE="nvidia"
            echo -e "${GREEN}Detected NVIDIA GPU with Docker runtime${NC}"
        else
            echo -e "${RED}NVIDIA GPU detected but NVIDIA Docker runtime not available${NC}"
            echo -e "${YELLOW}Falling back to AMD configuration (CPU-only mode)${NC}"
            COMPOSE_FILE="docker-compose.amd.yml"
            GPU_TYPE="amd-cpu-fallback"
        fi
    elif [ "$DETECTED_GPU" == "amd" ]; then
        COMPOSE_FILE="docker-compose.amd.yml"
        GPU_TYPE="amd"
        echo -e "${GREEN}Detected AMD GPU${NC}"
    else
        echo -e "${YELLOW}No GPU detected or unknown GPU type. Using AMD configuration (CPU-only)${NC}"
        COMPOSE_FILE="docker-compose.amd.yml"
        GPU_TYPE="cpu"
    fi
fi

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}Error: Compose file $COMPOSE_FILE not found${NC}"
    echo "Please make sure you have both docker-compose.amd.yml and docker-compose.nvidia.yml"
    exit 1
fi

# Create necessary directories
mkdir -p data/elasticsearch
mkdir -p data/ollama

echo -e "${GREEN}Starting services with $COMPOSE_FILE...${NC}"

# Build and start services
docker compose -f "$COMPOSE_FILE" build 
docker compose -f "$COMPOSE_FILE" up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to start...${NC}"

# Wait for Elasticsearch to be healthy
echo "Waiting for Elasticsearch..."
for i in {1..30}; do
    if curl -s -u elastic:changeme http://localhost:9200 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Elasticsearch is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Elasticsearch failed to start within 5 minutes${NC}"
        docker compose -f "$COMPOSE_FILE" logs elasticsearch
        exit 1
    fi
    sleep 10
done

# Wait for Ollama to be healthy
echo "Waiting for Ollama..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Ollama failed to start within 5 minutes${NC}"
        docker compose -f "$COMPOSE_FILE" logs ollama
        exit 1
    fi
    sleep 10
done

# Pull Ollama models with error handling
echo -e "${YELLOW}Pulling Ollama models...${NC}"

MODELS=("nomic-embed-text" "llama2:7b", "llama2-uncensored:7b")

for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    if docker exec ollama ollama pull "$model"; then
        echo -e "${GREEN}✓ Successfully pulled $model${NC}"
    else
        echo -e "${RED}✗ Failed to pull $model${NC}"
        echo "This might be normal if the model is already downloaded."
    fi
done

# Final health check
echo -e "${YELLOW}Performing final health checks...${NC}"

# Check if RAG app is running
if docker ps | grep rag-app > /dev/null; then
    echo -e "${GREEN}✓ RAG application is running${NC}"
else
    echo -e "${RED}✗ RAG application is not running${NC}"
    docker compose -f "$COMPOSE_FILE" logs rag-app
fi

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${YELLOW}Access URLs:${NC}"
echo -e "  Application:    http://localhost:7860"
echo -e "  Elasticsearch:  http://localhost:9200"
echo -e "  Kibana:         http://localhost:5601"
echo -e "  Ollama API:     http://localhost:11434"
echo ""
echo -e "${YELLOW}GPU Configuration: ${GPU_TYPE}${NC}"
echo -e "${YELLOW}To view logs: docker compose -f $COMPOSE_FILE logs -f${NC}"
echo -e "${YELLOW}To stop: docker compose -f $COMPOSE_FILE down${NC}"