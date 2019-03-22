FROM tensorflow/tensorflow:latest-gpu

# Install packages
RUN apt-get update &&\
    apt-get install vim -y &&\
    apt-get install git -y &&\
    apt-get install python-pip -y &&\
    apt-get install python-dev -y &&\
    apt-get install build-essential -y &&\
    pip install librosa --user &&\
    pip install keras==2.2.4 --user &&\
    pip install tqdm	


# Make a new working directory
RUN mkdir /monte-carlo-denoiser

# Switch to new working directory
WORKDIR /monte-carlo-denoiser
