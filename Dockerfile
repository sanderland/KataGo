from debian:buster

RUN apt-get update
RUN apt-get install -y g++ git cmake libeigen3-dev ocl-icd-opencl-dev libzip-dev

COPY . /KataGo
WORKDIR /KataGo/cpp
RUN rm CMakeCache.txt || true

RUN cmake . -DUSE_BACKEND=OPENCL && make && strip katago && cp katago katago-opencl
RUN cmake . -DUSE_BACKEND=EIGEN && make && strip katago && cp katago katago-eigen
