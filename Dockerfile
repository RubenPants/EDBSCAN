# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE=continuumio/miniconda3:latest

# # 1. Create COMPILE image

FROM $BASE_IMAGE AS compile-image

# Allow cloning private GitLab repositories.
ARG GITLAB_CI_TOKEN
RUN if [ -n "$GITLAB_CI_TOKEN" ]; then \
    git config --global url."https://gitlab-ci-token:$GITLAB_CI_TOKEN@gitlab.com/".insteadOf "ssh://git@gitlab.com/"; \
    else \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan gitlab.com >> ~/.ssh/known_hosts; \
    fi

# Install compilers for certain pip requirements.
RUN (apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*) || \
    (dnf install -y gcc gcc-c++ make && dnf clean all && rm -rf /var/cache/yum)

# Install conda environments. Minification inspired by [1].
# [1] https://jcrist.github.io/conda-docker-tips.html
COPY environment.*.yml ./
RUN --mount=type=ssh pip install conda-merge && \
    conda-merge environment.run.yml environment.dev.yml > environment.yml && \
    conda install mamba --channel conda-forge --yes && \
    mamba env create --file environment.yml && \
    mamba env create --file environment.run.yml && \
    conda clean --yes --all --force-pkgs-dirs --quiet && \
    cd /opt/conda/envs/edbscan-run-env/lib/python*/site-packages && du --max-depth=3 --threshold=5M -h | sort -h && cd - && \
    find /opt/conda/ -follow -type d -name '__pycache__' -exec rm -rf {} + && \
    find /opt/conda/ -follow -type d -name 'examples' -not -path '*tensorflow*' -exec rm -rf {} + && \
    find /opt/conda/ -follow -type d -name 'tests' -exec rm -rf {} + && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyo' -delete && \
    cd /opt/conda/envs/edbscan-run-env/lib/python*/site-packages && du --max-depth=3 --threshold=5M -h | sort -h && cd -

# # 2. Create CI image

FROM $BASE_IMAGE AS ci-image

# Configure Python.
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Copy the conda environment from the compile-image.
COPY --from=compile-image /root/.conda/ /root/.conda/
COPY --from=compile-image /opt/conda/ /opt/conda/

# Activate conda environment.
ENV PATH /opt/conda/envs/edbscan-env/bin:$PATH
RUN echo "source activate edbscan-env" >> ~/.bashrc

# # 3. Create application image

FROM $BASE_IMAGE AS app-image

# Automatically activate conda environment when opening a bash shell with `/bin/bash`.
WORKDIR /app/src/
ENV PYTHONPATH /app/src/:$PYTHONPATH
ENV PATH /opt/conda/envs/edbscan-run-env/bin:$PATH
RUN echo "source activate edbscan-run-env" >> ~/.bashrc

# Create Docker entrypoint.
RUN printf '#!/usr/bin/env bash\n\
    \n\
    set -e\n\
    \n\
    function run_dev {\n\
    echo "Running Development Server on 0.0.0.0:8000"\n\
    uvicorn "edbscan.api:app" --reload --log-level debug --host 0.0.0.0\n\
    }\n\
    \n\
    function run_serve {\n\
    echo "Running Production Server on 0.0.0.0:8000"\n\
    gunicorn --bind 0.0.0.0 --workers=2 --timeout 30 --graceful-timeout 10 --keep-alive 10 --worker-tmp-dir /dev/shm --access-logfile - --log-file - -k uvicorn.workers.UvicornWorker "edbscan.api:app"\n\
    }\n\
    \n\
    case "$1" in\n\
    dev)\n\
    run_dev\n\
    ;;\n\
    serve)\n\
    run_serve\n\
    ;;\n\
    bash)\n\
    /bin/bash "${@:2}"\n\
    ;;\n\
    esac\n\
    ' > /usr/local/bin/entrypoint.sh && chmod ugo+x /usr/local/bin/entrypoint.sh

# Configure application.
ARG PORT=8000
EXPOSE $PORT
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]

# Add source code to the `WORKDIR`.
COPY src .

# The following environment variables are supplied as build args at build time [1].
# [1] https://docs.docker.com/docker-hub/builds/advanced/
ARG ENVIRONMENT
ENV ENVIRONMENT $ENVIRONMENT
ARG SOURCE_BRANCH
ENV SOURCE_BRANCH $SOURCE_BRANCH
ARG SOURCE_COMMIT
ENV SOURCE_COMMIT $SOURCE_COMMIT
ARG SOURCE_TIMESTAMP
ENV SOURCE_TIMESTAMP $SOURCE_TIMESTAMP
