FROM tensorflow/tensorflow:latest-gpu

# Install packages
RUN apt-get -y update &&\
    apt-get -y install vim &&\
    apt-get -y install git &&\
    apt-get -y install python-pip &&\
    apt-get -y install python-dev &&\
    apt-get -y install build-essential &&\
    apt-get -y install python-tk &&\
    pip install librosa --user &&\
    pip install keras==2.2.4 --user &&\
    pip install tqdm


# Make a new working directory
RUN mkdir /monte-carlo-denoiser

# Switch to new working directory
WORKDIR /monte-carlo-denoiser

RUN cd custom_ops && ./compile.sh & cd ..
