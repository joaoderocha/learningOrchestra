FROM rayproject/ray-ml:latest-cpu


RUN sudo apt-get update && sudo apt-get install -y wget

RUN sudo apt-get -y install build-essential

RUN sudo apt-get -y install cmake gcc g++

ENV DOCKERIZE_VERSION v0.6.1

RUN sudo wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && sudo tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && sudo rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz


COPY start.sh .

RUN HOROVOD_WITH_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 pip install 'horovod[ray,tensorflow,keras]'