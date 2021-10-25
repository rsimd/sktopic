FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
        # if a CI/CD system is expected to pass in these arguments
	    # the dockerfile should be modified accordingly
	        JAXLIB_VERSION=0.1.69

# upgrade python version
WORKDIR /opt
RUN apt-get update && apt-get install wget -y && \
    apt-get install -y build-essential libsqlite3-dev sqlite3 bzip2 libbz2-dev \
    	    zlib1g-dev libssl-dev openssl libgdbm-dev libgdbm-compat-dev liblzma-dev libreadline-dev \
    	    libncursesw5-dev libffi-dev uuid-dev && \
    wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz && \
    tar xzf Python-3.9.7.tgz

RUN cd Python-3.9.7 && \
    ./configure --enable-optimizations

RUN cd /opt/Python-3.9.7 && \
    make altinstall && \
    update-alternatives --install /usr/bin/python python /opt/Python-3.9.7/python 0

RUN wget  https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN echo 'alias pip="pip3.9"' >> ~/.bashrc
RUN pip install -U pip && pip install poetry
WORKDIR /root/.ipython/profile_default/startup
RUN wget https://gist.githubusercontent.com/knowsuchagency/f7b2203dd613756a45f816d6809f01a6/raw/c9c1b7ad9fa25a57ee1903d755d5525894ffa411/typecheck.py
WORKDIR /workdir
RUN /usr/local/bin/poetry install
RUN /usr/local/bin/poetry shell