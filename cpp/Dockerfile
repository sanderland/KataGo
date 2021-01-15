from debian:buster

RUN apt-get update
RUN apt-get install -y g++ git cmake libeigen3-dev ocl-icd-opencl-dev libzip-dev

WORKDIR /KataGo
COPY . .
RUN rm CMakeCache.txt || true

RUN cmake . -DUSE_BACKEND=OPENCL
RUN make
RUN strip katago
RUN COPY katago katago-opencl

RUN cmake . -DUSE_BACKEND=EIGEN
RUN make
RUN strip katago
RUN COPY katago katago-eigen

