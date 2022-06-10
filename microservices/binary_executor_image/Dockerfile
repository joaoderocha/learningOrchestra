FROM python:3.7-slim

WORKDIR /usr/src/binary_executor
COPY . /usr/src/binary_executor

RUN sudo apt-get update

RUN sudo apt-get -y install build-essential

RUN sudo apt-get -y install cmake gcc g++

RUN pip install -r requirements.txt

RUN pip install -U "ray[default]"

RUN HOROVOD_WITH_GLOO=1 HOROVOD_WITHOUT_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip install 'horovod[ray,tensorflow,keras]'



ENV MICROSERVICE_IP "0.0.0.0"
ENV MICROSERVICE_PORT 5008
ENV BINARY_VOLUME_PATH "/binaries"
ENV MODELS_VOLUME_PATH "/models"
ENV TRANSFORM_VOLUME_PATH "/transform"
ENV CODE_EXECUTOR_VOLUME_PATH "/code_executions"

CMD ["python", "server.py"]