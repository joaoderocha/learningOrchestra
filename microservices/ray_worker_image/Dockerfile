FROM rayproject/ray:latest-cpu


RUN sudo apt-get update && sudo apt-get install -y wget

ENV DOCKERIZE_VERSION v0.6.1

RUN sudo wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && sudo tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && sudo rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz

COPY start.sh .
