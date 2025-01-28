# Use a base image with Conda already installed
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy only environment.yml first (for better caching)
COPY environment.yml /app/environment.yml

COPY requirements.txt /app/requirements.txt

# Ensure the environment is clean (remove old env if exists, but ignore error)
RUN conda clean --all --yes && \
    (conda env remove -n asr_v2 || true) && \
    conda env create -f environment.yml

# Switch shell to use the base shell (we will activate the environment later)
SHELL ["/bin/bash", "-c"]

# Now copy in all other source files (including requirements.txt, app.py, etc.)
COPY . /app

# (Optional) Install extra pip requirements if not included in environment.yml
RUN source activate asr_v2 && pip install --no-cache-dir -r requirements.txt

# Activate the environment and ensure it is set correctly
RUN echo "conda activate asr_v2" >> ~/.bashrc

# Set the default command to activate the environment and run the app
CMD ["/bin/bash", "-c", "source activate asr_v2 && python app.py"]

# (Optional) Instructions for running the container with GPU and environment activation:
# docker run --gpus all -it -p 7860:7860 ichigo /bin/bash
# Once inside the container, activate the environment and run the app:
# conda activate asr_v2
# python app.py
