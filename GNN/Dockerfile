# Common

# Potato
#FROM --platform=amd64 nvidia/cuda:11.3.1-runtime-ubuntu18.04
FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04
# RCI
# FROM --platform=amd64 nvidia/cuda:12.4.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && apt-add-repository ppa:swi-prolog/stable \
    && apt-get update \
    && apt-get install -y make gcc git curl \
    zlib1g zlib1g-dev python3-pip libssl-dev \
    bzip2 libbz2-dev sqlite3 libsqlite3-dev \
    libreadline-dev libffi-dev liblzma-dev \
    wget lmdb-utils python-dev build-essential \
    graphviz graphviz-dev cmake swi-prolog \
    texlive texlive-latex-extra  texlive-fonts-recommended \
    texlive-fonts-recommended \
    texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended texlive-publishers  texlive-science  texlive-xetex \
    dvipng cm-super dvipng ghostscript cm-super

ARG USER_ID=1000
ARG GROUP_ID=1000
# RUN (addgroup --gid $GROUP_ID app || true) \
#     && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID app
RUN (addgroup --gid 1000 app || true) \
    && adduser --disabled-password --gecos '' --uid 1000 --gid 1000 app
USER app

ENV HOME /home/app
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ARG PYTHON_VERSION

# RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv \
#     && pyenv install $PYTHON_VERSION \
#     && pyenv global $PYTHON_VERSION \
#     && pip install --upgrade "pip<24.1" \
#     && pip install pipenv \
#     && curl -sSL https://install.python-poetry.org | python -
# 修改后
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pip install --upgrade "pip<24.1" \
    && pip install pipenv \
    && mkdir -p $HOME/.local/bin \
    && pip install poetry

ENV PATH="${PATH}:${HOME}/.local/bin"
ENV PATH $PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH

WORKDIR /app
ENV PYTHONPATH /app
ENV PYTHONDONTWRITEBYTECODE 1

# Standard stuff

RUN poetry config virtualenvs.create false

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN poetry install

# A100 compatible torch installation

RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install pytorch-lightning==1.7.7
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Some of DiGress stuff

USER root

# RUN apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25 \
#     && add-apt-repository "deb [ arch=amd64 ] https://downloads.skewed.de/apt bionic main" \
#     # && add-apt-repository "deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main" \
#     && apt-get update \
#     && apt-get install -y python3-graph-tool

USER app

RUN pip install git+https://github.com/igor-krawczuk/mini-moses@master

# GraphINVENT stuff

RUN pip install h5py

# Microsoft molecule gen

RUN pip install molecule-generation

# Conda

USER root

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
ENV PATH=$PATH:$CONDA_DIR/bin
RUN chmod -R a+w $CONDA_DIR

# DEG

COPY data_efficient_grammar/environment.yml data_efficient_grammar/environment.yml
COPY data_efficient_grammar/retro_star/environment.yml data_efficient_grammar/retro_star/environment.yml
COPY data_efficient_grammar/retro_star/packages data_efficient_grammar/retro_star/packages
RUN chmod -R a+w data_efficient_grammar/

USER app

RUN conda env create -f data_efficient_grammar/environment.yml
RUN conda env create -f data_efficient_grammar/retro_star/environment.yml
RUN pip install -e data_efficient_grammar/retro_star/packages/mlp_retrosyn && \
    pip install -e data_efficient_grammar/retro_star/packages/rdchiral

# Reinvent

USER root

COPY reinvent-randomized/environment.yml reinvent-randomized/environment.yml
RUN chmod -R a+w reinvent-randomized/

USER app

RUN conda env create -f reinvent-randomized/environment.yml

# Paccmann

USER root

COPY paccmann_chemistry/examples/conda.yml paccmann_chemistry/examples/conda.yml
RUN chmod -R a+w paccmann_chemistry/

USER app

RUN conda env create -f paccmann_chemistry/examples/conda.yml

# rgcvae

# RUN pip install Cython

# USER root

# COPY rgcvae/rgivae_env.yml rgcvae/rgivae_env.yml
# RUN chmod -R a+w rgcvae/

# RUN conda env create -f rgcvae/rgivae_env.yml

# USER app

# ccgvae

USER root

COPY ccgvae/ccgvae_env.yml ccgvae/ccgvae_env.yml
RUN chmod -R a+w ccgvae/

RUN conda env create -f ccgvae/ccgvae_env.yml

USER app

# End

ENV PYTHONPATH "/app:/app/DiGress:/app/GraphINVENT:/app/data_efficient_grammar:/app/reinvent-randomized:/app/Molecule-RNN:/app/paccmann_chemistry"

# Temp

RUN pip install selfies

RUN pip install torchmetrics==0.11.4

RUN pip install pydantic

# Set working directory
WORKDIR /app
ENV PYTHONPATH "/app:/app/DiGress:/app/GraphINVENT:/app/data_efficient_grammar:/app/reinvent-randomized:/app/Molecule-RNN:/app/paccmann_chemistry"

# Install additional packages
RUN pip install -e data_efficient_grammar/retro_star/packages/mlp_retrosyn && \
    pip install -e data_efficient_grammar/retro_star/packages/rdchiral
##pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html