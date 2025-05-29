# Use an NVIDIA CUDA base image. Choose a version compatible with llama-cpp-python's requirements
# and your g4dn.xlarge instance (Tesla T4 typically supports CUDA 11.x, 12.x).
# Check llama-cpp-python's documentation for recommended CUDA versions.
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
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
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python and pip
# Use -sf to force creation of symbolic links, overwriting if they exist
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PDM
RUN pip install --no-cache-dir pdm

# Install uv
RUN pip install --no-cache-dir uv

# Copy project configuration first
COPY pyproject.toml ./
# pdm.lock is intentionally not copied to allow PDM to resolve based on pyproject.toml in the build env

# Set environment variables for CUDA compilation and general build
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
# Set LD_LIBRARY_PATH in a single layer to ensure proper expansion.
# This includes CUDA paths and GCC library paths as suggested by the GitHub issue.
RUN LLAMA_CPP_PYTHON_VERSION="0.3.9" && \
    LLAMA_CPP_PYTHON_WHEEL_URL="https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.9-cu126-AVX2-linux-20250525/llama_cpp_python-0.3.9-cp310-cp310-linux_x86_64.whl" && \
    \
    # Set environment variables for this build step.
    # These are often for source builds, but we keep them as per user's request to follow GitHub issue contexts.
    export CC=/usr/bin/gcc && \
    export CUDA_PATH=/usr/local/cuda && \
    export CUDA_CXX=/usr/local/cuda/bin/nvcc && \
    export CXX=/usr/bin/g++ && \
    # Ensure GCC libs and CUDA stubs are in LD_LIBRARY_PATH for this RUN command's environment
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/lib/gcc/$(gcc -dumpmachine)/$(gcc -dumpversion):${LD_LIBRARY_PATH}" && \
    export FORCE_CMAKE=1 && \
    # For g4dn (Tesla T4), CUDA compute capability is 7.5. This might need adjustment if your target HW is different,
    # but for a cu126 wheel, the architectures should be baked into the wheel. Kept for consistency with user's file.
    export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF" && \
    \
    echo "--- Environment for llama-cpp-python install step ---" && \
    echo "LLAMA_CPP_PYTHON_WHEEL_URL: ${LLAMA_CPP_PYTHON_WHEEL_URL}" && \
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}" && \
    echo "CC: ${CC}, CXX: ${CXX}" && \
    echo "CUDACXX: ${CUDACXX}" && \
    echo "CMAKE_ARGS: ${CMAKE_ARGS}" && \
    echo "FORCE_CMAKE: ${FORCE_CMAKE}" && \
    echo "----------------------------------------------------" && \
    \
    # Install llama-cpp-python directly from the specified wheel URL.
    # Removed --index-url, --extra-index-url, and --index-strategy as they are not used for direct wheel URLs.
    # The [server] extra is not used here; PDM will install FastAPI/Uvicorn from pyproject.toml.
    uv pip install --system --upgrade --no-cache --force-reinstall "${LLAMA_CPP_PYTHON_WHEEL_URL}" && \
    \
    # Verify llama-cpp-python installation
    echo "Verifying llama-cpp-python installation..." && \
    python -m llama_cpp --version && \
    (python -c "from llama_cpp import Llama; print('Successfully imported Llama from llama_cpp')" || \
     (echo "Llama import test failed, attempting llama_cpp_init..." && python -c "from llama_cpp import llama_cpp_init; llama_cpp_init(); print('llama_cpp_init successful')")) && \
    echo "llama-cpp-python installation step completed."

# Install remaining project dependencies using PDM.
# llama-cpp-python should already be installed by uv pip.
RUN pdm install -vv --prod --no-editable

# Copy the rest of your application source code
# If api.py is in src/, copy the src directory
COPY src/ ./src/

# Create a directory for models inside the container
RUN mkdir -p /models

# Expose the port the FastAPI app will run on
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# We use pdm run to execute commands using the project's environment
CMD ["pdm", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]