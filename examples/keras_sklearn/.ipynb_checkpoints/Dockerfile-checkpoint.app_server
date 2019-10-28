FROM continuumio/miniconda3

USER root

# App requirements
RUN mkdir -p /root/project
ADD requirements.txt /root/project
WORKDIR /root/project
RUN pip install -r requirements.txt

