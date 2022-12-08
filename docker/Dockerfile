ARG CUDA_DOCKER_VERSION=11.7.1-cudnn8-devel-ubuntu22.04

FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install curl -y
RUN apt-get install git -y
RUN apt-get install wget -y
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV TZ Asia/Seoul

RUN apt-get install libsndfile1-dev -y --no-install-recommends

RUN git clone -b v2022 https://github.com/kotechnia/sound-classification.git /root/workspace
#RUN mkdir -p /root/workspace
WORKDIR /root/workspace

RUN pip install -r /root/workspace/requirements.txt
