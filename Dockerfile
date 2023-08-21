FROM continuumio/miniconda3:23.5.2-0

RUN apt-get update -qq && \
    apt-get install -qq -y && \
    apt-get -y install wget unzip

RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip && \
    unzip fiji-linux64.zip

RUN pip install --upgrade granny
