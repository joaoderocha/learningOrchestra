FROM python:3.7.7-slim

WORKDIR /usr/src/code_executor
COPY . /usr/src/code_executor

RUN apt-get update

RUN apt-get -y install build-essential

RUN apt-get -y install cmake gcc g++ curl

RUN pip install -r requirements.txt

RUN HOROVOD_WITH_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 pip install 'horovod[ray,tensorboard,keras]'

RUN ray start --address='10.182.0.21:6379' --redis-password='5241590000000000' --object-manager-port=8076 --num-cpus=0 --num-gpus=0

ENV MICROSERVICE_IP "0.0.0.0"
ENV MICROSERVICE_PORT 5009
ENV EXPLORE_VOLUME_PATH "/explore"
ENV TRANSFORM_VOLUME_PATH "/transform"
ENV BINARY_VOLUME_PATH "/binaries"
ENV MODELS_VOLUME_PATH "/models"
ENV CODE_EXECUTOR_VOLUME_PATH "/code_executions"
ENV DATASET_VOLUME_PATH "/datasets"

CMD ["python", "server.py"]
