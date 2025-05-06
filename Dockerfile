FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y dos2unix

RUN dos2unix setup.sh && chmod +x setup.sh && ./setup.sh
