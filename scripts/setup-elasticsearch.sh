#!/bin/bash
# setup-elasticsearch.sh

echo "Waiting for Elasticsearch to start..."
until curl -s http://elasticsearch:9200 >/dev/null; do
  sleep 5
done

echo "Setting kibana_system password..."
curl -X POST -u elastic:changeme "http://elasticsearch:9200/_security/user/kibana_system/_password" \
  -H "Content-Type: application/json" \
  -d '{"password": "changeme"}'

echo "Kibana system password set successfully!"