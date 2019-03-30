FROM tensorflow/tensorflow:1.1.0-gpu
# Tensorflow is not required, but to save disk space we reuse this base image

RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    libtbb-dev \
    libjpeg-dev \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
    pyexr \
    click

ADD tungsten /tungsten
WORKDIR /tungsten
RUN ./setup_builds.sh
WORKDIR /tungsten/build/release
RUN make -j 10

ENV TUNGSTEN_BINARY /tungsten/build/release/tungsten

ADD tungsten.py /tungsten.py
RUN chmod +x /tungsten.py

VOLUME /scene
WORKDIR /scene

ENTRYPOINT ["/tungsten.py"]
