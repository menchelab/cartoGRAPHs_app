#FROM openanalytics/r-base
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# system libraries of general use
RUN apt-get update && apt-get install -y \
    sudo \
    pandoc \
    pandoc-citeproc \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    libxt-dev \
    libssl-dev \
    libssh2-1-dev \
    libssl1.1 \
    python3 \
    python3-pip

USER root

#
# install dependencies of the cartoGRAPHs app
COPY Requirements.txt Requirements.txt
RUN python3 -m pip install -r Requirements.txt
#RUN conda install -c conda-forge mkl-service==2.3.0
#RUN conda install -c intel mkl_random==1.2.1

# copy the app to the image
RUN mkdir cartoGRAPHs
COPY cartoGRAPHs_app/* cartoGRAPHs/

EXPOSE 3838
EXPOSE 8050 

WORKDIR /cartoGRAPHs
CMD ["python3", "app.py"]
