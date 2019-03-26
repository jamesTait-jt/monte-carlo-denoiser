FROM tensorflow/tensorflow:latest-gpu

# Install packages
RUN apt-get -y update &&\
    apt-get -y install vim -y &&\
    apt-get -y install git -y &&\
    apt-get -y install python-pip -y &&\
    apt-get -y install python-dev -y &&\
    apt-get -y install build-essential -y &&\
    apt-get -y install python-tk &&\
    pip install librosa --user &&\
    pip install keras==2.2.4 --user &&\
    pip install tqdm &&\
    pip install python-tk


# Make a new working directory
RUN mkdir /monte-carlo-denoiser

# Switch to new working directory
WORKDIR /monte-carlo-denoiser
