FROM mcr.microsoft.com/devcontainers/miniconda:0-3

ARG USERNAME="vscode"
ENV USER_HOME=/home/${USERNAME}
ARG ENV_NAME="evagg"

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/

# Run remaining commands as non-root user.
USER ${USERNAME}

# Force ${USER_HOME}/.bashrc to update.
RUN /opt/conda/bin/conda init bash

# Build the environment.
RUN /opt/conda/bin/conda config --add channels conda-forge \
    && /opt/conda/bin/conda install -n base conda-libmamba-solver \
    && /opt/conda/bin/conda config --set solver libmamba \
    && /opt/conda/bin/conda env create -n ${ENV_NAME} -f /tmp/conda-tmp/environment.yml \
    && rm -rf /tmp/conda-tmp

# Tilde is ok because we've set the user above.
RUN echo "conda activate ${ENV_NAME}" >> ${USER_HOME}/.bashrc \
    && echo "poetry install --no-interaction" >> ${USER_HOME}/.bashrc

