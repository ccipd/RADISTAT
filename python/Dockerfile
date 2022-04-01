FROM python:3.7.13-buster

COPY requirements.txt requirements.txt

RUN pip install numpy
RUN pip install scikit-build
RUN pip install -r requirements.txt

# In a terminal inside this folder, run:
#   docker build . -t radistat:latest
#   docker run --mount type=bind,source="$(pwd)",target=/workdir --rm -it radistat:latest /bin/bash
