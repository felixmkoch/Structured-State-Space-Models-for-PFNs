FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
WORKDIR /app
COPY . .

RUN DEBIAN_FRONTEND=noninteractive ./setup.sh
