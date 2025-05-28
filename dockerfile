# Use an NVIDIA CUDA base image. Choose a version compatible with llama-cpp-python's requirements
# and your g4dn.xlarge instance (Tesla T4 typically supports CUDA 11.x, 12.x).
# Check llama-cpp-python's documentation for recommended CUDA versions.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# Install Python, pip, git, cmake, and other build essentials
# Using python3.10 as an example, align with your pyproject.toml's requires-python if specified
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python and pip
# Use -sf to force creation of symbolic links, overwriting if they exist
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PDM
RUN pip install --no-cache-dir pdm

# Copy project configuration
COPY pyproject.toml ./

# Set CMAKE_ARGS to build llama-cpp-python with CUDA support.
# This must be set *before* pdm install if llama-cpp-python is compiled during install.
ENV CMAKE_ARGS="-DLLAMA_CUDA=ON"
ENV FORCE_CMAKE=1

# Install project dependencies using PDM.
# PDM will use pyproject.toml. If pdm.lock is not found (because we removed the COPY for it),
# PDM will resolve dependencies and create pdm.lock within the build environment.
RUN pdm install --prod --no-editable

# Copy the rest of your application source code
# If api.py is in src/, copy the src directory
COPY src/ ./src/

# Create a directory for models inside the container
RUN mkdir -p /models

# Expose the port the FastAPI app will run on
EXPOSE 8000

# Set environment variables for model location (can be overridden at runtime if needed)
# These should ideally be passed during `docker run` or by the EC2 user data script
# ENV MODEL_S3_BUCKET="your-s3-bucket-name-for-models" # Set at runtime
# ENV MODEL_S3_KEY="your-model-file.gguf"             # Set at runtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run when the container starts
# PDM will ensure that the command is run within the project's virtual environment
# We use pdm run to execute commands using the project's environment
CMD ["pdm", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
