#!/bin/bash

# Try to stop using the correct compose file
if [ -f "docker-compose.amd.yml" ]; then
    docker compose -f docker-compose.amd.yml down
fi

if [ -f "docker-compose.nvidia.yml" ]; then
    docker compose -f docker-compose.nvidia.yml down
fi

echo "All services stopped"