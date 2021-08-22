FROM nvcr.io/nvidia/pytorch:21.02-py3
SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING "utf-8"
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8

RUN apt-get update \
    && apt-get install -y python3 python3-pip python3-llvmlite\
    && apt-get install -y tzdata 
ENV TZ=Asia/Tokyo 
RUN apt-get install -y mpich git cron curl make
RUN apt install -y ffmpeg sox

COPY requirements.txt ./
# RUN pip3 install -U pip && \
RUN ls
RUN pip3 install -r ./requirements.txt 
    # pip3 install tensorflow==1.2.1