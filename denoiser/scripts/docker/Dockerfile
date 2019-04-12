FROM tensorflow/tensorflow:latest-gpu-py3

# Install packages
RUN apt-get -y update &&\
    apt-get -y install vim &&\
    apt-get -y install git &&\
    apt-get -y install python3-pip &&\
    apt-get -y install python3-dev &&\
    apt-get -y install build-essential &&\
    apt-get -y install python-tk &&\
    apt-get -y install libopenexr-dev &&\
    pip3 install librosa --user &&\
    pip3 install keras==2.2.4 --user &&\
    pip3 install tqdm --user &&\
    pip3 install OpenEXR --user &&\
    pip3 install Imath --user &&\
    pip3 install matplotlib --user &&\
    pip3 install numpy --user 


# Make a new working directory
RUN mkdir /monte-carlo-denoiser

# Switch to new working directory
WORKDIR /monte-carlo-denoiser

RUN cd custom_ops && ./compile.sh & cd ..
