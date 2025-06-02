# Use an NVIDIA CUDA base image.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# Install Python, pip, git, and other build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    # build-essential might be needed for some PDM build backends or if torch needs to compile parts
    build-essential \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PDM
RUN pip install --no-cache-dir pdm

# Copy project configuration first
COPY pyproject.toml pdm.lock ./ 
# Copy pdm.lock if you want reproducible dependency versions. 
# If not, PDM will resolve from pyproject.toml. Given you're building in CodeBuild,
# having a lock file is good practice. If you don't have one yet, PDM will create it.

# Set environment variables for CUDA (PyTorch should pick these up)
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install project dependencies using PDM.
# This will install torch, transformers, accelerate, bitsandbytes from pyproject.toml.
# Ensure your base image (nvidia/cuda:12.6.0) has compatible drivers/toolkit for the PyTorch version.
# PDM uses pip under the hood, which should pull CUDA-enabled torch wheels if available.
RUN pdm install -vv --prod --no-editable

# Copy the rest of your application source code
COPY src/ ./src/

# Create a directory for models inside the container
RUN mkdir -p /models

# Expose the port the FastAPI app will run on
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# We use pdm run to execute commands using the project's environment
CMD ["pdm", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]