FROM horovod/horovod-ray:0.24.3

WORKDIR /usr/src/binary_executor
COPY . /usr/src/binary_executor

RUN sudo apt-get install -y supervisor

RUN pip install -r requirements.txt


RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV MICROSERVICE_IP "0.0.0.0"
ENV MICROSERVICE_PORT 5008
ENV BINARY_VOLUME_PATH "/binaries"
ENV MODELS_VOLUME_PATH "/models"
ENV TRANSFORM_VOLUME_PATH "/transform"
ENV CODE_EXECUTOR_VOLUME_PATH "/code_executions"

CMD ["/usr/bin/supervisord"]