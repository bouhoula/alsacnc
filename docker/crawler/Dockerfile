FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

# Update ubuntu and setup conda
# adapted from: https://hub.docker.com/r/conda/miniconda3/dockerfile
RUN sed -i'' 's/archive\.ubuntu\.com/us\.archive\.ubuntu\.com/' /etc/apt/sources.list
RUN apt-get clean -qq \
    && rm -r /var/lib/apt/lists/* -vf \
    && apt-get clean -qq \
    && apt-get update -qq \
    && apt-get upgrade -qq \
    # git and make for `npm install`, wget for `install-miniconda`
    && apt-get install wget git make gcc g++ -qq \
    # deps to run firefox inc. with xvfb
    && apt-get install firefox xvfb libgtk-3-dev libasound2 libdbus-glib-1-2 libpci3 -qq


ENV HOME /opt
COPY docker/crawler/install-mamba.sh .
COPY docker/crawler/mamba.sh .
RUN ./install-mamba.sh
ENV PATH $HOME/mamba/bin:$PATH


# Install OpenWPM
WORKDIR /opt
RUN mkdir crawler
WORKDIR /opt/crawler
RUN git clone --depth 1 --branch v0.23.0 https://github.com/openwpm/OpenWPM.git openwpm
WORKDIR /opt/crawler/openwpm
RUN ./install.sh
ENV PATH $HOME/mamba/envs/openwpm/bin:$PATH


# Move the firefox binary away from the /opt/OpenWPM root so that it is available if
# we mount a local source code directory as /opt/OpenWPM
RUN mv firefox-bin /opt/firefox-bin
ENV FIREFOX_BINARY /opt/firefox-bin/firefox-bin

# Clone Tracker Radar repo
WORKDIR /opt
RUN git clone --depth 1 https://github.com/duckduckgo/tracker-radar.git

# Install python dependencies
WORKDIR /opt/crawler
COPY requirements requirements
RUN apt-get install libpq-dev -qq
RUN pip install -r requirements/shared_requirements.txt
RUN pip install -r requirements/crawler_requirements.txt

RUN for model_name in da_core_news_sm de_core_news_sm el_core_news_sm en_core_web_sm es_core_news_sm fi_core_news_sm \
     fr_core_news_sm it_core_news_sm nl_core_news_sm pl_core_news_sm pt_core_news_sm sv_core_news_sm; do  \
    python -m spacy download $model_name; \
    done

# Clone Consent-o-matic repo
RUN git clone --depth 1 https://github.com/cavi-au/Consent-O-Matic.git consentomatic

# Set PYTHONPATH
ENV PYTHONPATH /opt/crawler:/opt/crawler/openwpm

# Remove excessive logging from OpenWPM
RUN sed -i '/self.logger.info/{N;/get_configuration_string/{N;N;N;d;};}' openwpm/openwpm/task_manager.py

# Downlaod models
RUN apt-get install unzip
RUN wget https://zenodo.org/records/7638513/files/models.zip &&\
    unzip models.zip && rm models.zip && rm -r __MACOSX

# Copy project
COPY . .

CMD ["python", "cookie_crawler/run_crawler.py"]
