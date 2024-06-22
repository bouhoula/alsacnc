FROM python:3.9

WORKDIR /opt
RUN mkdir repo

WORKDIR /opt/repo
COPY requirements requirements
RUN pip install -r requirements/shared_requirements.txt
RUN pip install -r requirements/ml_requirements.txt

ENV PYTHONPATH /opt/repo

# Downlaod data
RUN apt-get install unzip
RUN wget https://zenodo.org/records/12209965/files/consentomatic-reject-tranco05May-20210514_081933.json -P data
RUN wget https://zenodo.org/records/12209965/files/tranco_05May_20210510_201615.json -P data

COPY . .

