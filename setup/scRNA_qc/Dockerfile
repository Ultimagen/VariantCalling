FROM python:3.11-bullseye as build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH=/usr/bin/poetry/bin:$PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
    POETRY_HOME=/usr/bin/poetry

RUN curl -sSL https://install.python-poetry.org > ./install-poetry.py && \
    python ./install-poetry.py && \
    rm ./install-poetry.py

# Create virtualenv for deployment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . .
COPY ./setup/scRNA_qc/pyproject.toml ./pyproject.toml
RUN poetry install --only main && \
    poetry run pip install .

################################################

FROM python:3.11-slim

ARG BUILD_TIMESTAMP
ARG BUILD_VERSION
ARG VERSION

ARG HOST_INSTALL_DIR=/install

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

LABEL com.ultimagenomics.build-timestamp=${BUILD_TIMESTAMP}
LABEL com.ultimagenomics.build-version=${BUILD_VERSION}
LABEL com.ultimagenomics.version=${VERSION}

# Install apt-get dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    libgomp1 \
    unzip \
    procps `# for monitoring script memory usage` \
    sysstat `# for monitoring script IO metrics` \
    ; \
    apt-get autoclean && rm -rf /var/lib/apt/lists/*

# Install STAR
RUN wget -O- https://github.com/alexdobin/STAR/archive/2.7.10a.tar.gz \
    | tar xfz - STAR-2.7.10a/bin/Linux_x86_64/STAR \
    && mv STAR-2.7.10a/bin/Linux_x86_64/STAR /usr/local/bin \
    && rm -rf STAR-2.7.10a

WORKDIR /app/VariantCalling
COPY --from=build /opt/venv /opt/venv
COPY . .
COPY ./setup/scRNA_qc/pyproject.toml ./pyproject.toml

# Include licenses
COPY ["Ultima UGVC - EULA.pdf", "./"]
COPY ["Third Party Licenses/star LICENSE.txt", "."]