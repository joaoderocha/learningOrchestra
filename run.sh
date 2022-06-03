#!/bin/bash

echo "learningOrchestra: a distributed machine learning processing tool"
echo "--------------------------------------------------------------------"
echo "Building the learningOrchestra microservice images..."
echo "--------------------------------------------------------------------"

docker build --tag spark_task microservices/spark_task_image
docker push 127.0.0.1:5050/spark_task

docker build --tag distributed_training_image microservices/distributed_training_image
docker push 127.0.0.1:5050/distributed_training_image


echo "--------------------------------------------------------------------"
echo "Adding the microservice images in docker daemon security exception..."
echo "--------------------------------------------------------------------"

echo '{
  "insecure-registries" : ["myregistry:5050"]
}
' > /etc/docker/daemon.json

echo "--------------------------------------------------------------------"
echo "Restarting docker service..."
echo "--------------------------------------------------------------------"

service docker restart

echo "--------------------------------------------------------------------"
echo "Deploying learningOrchestra in swarm cluster..."
echo "--------------------------------------------------------------------"

docker stack deploy --compose-file=docker-compose.yml microservice

echo "--------------------------------------------------------------------"
echo "Pushing the microservice images in local repository..."
echo "--------------------------------------------------------------------"

sleep 30


distributed_training_repository=127.0.0.1:5050/distributed_training


echo "--------------------------------------------------------------------"
echo "Pushing distributedTraining microservice image..."
echo "--------------------------------------------------------------------"
docker push $distributed_training_repository


echo "--------------------------------------------------------------------"
echo "Updating portainer agent microservice in each cluster node..."
echo "--------------------------------------------------------------------"
docker service update --image portainer/agent  microservice_agent

echo "--------------------------------------------------------------------"
echo "End."
echo "--------------------------------------------------------------------"
