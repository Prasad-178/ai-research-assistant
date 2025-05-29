# Use an NVIDIA CUDA base image. Choose a version compatible with llama-cpp-python's requirements
# and your g4dn.xlarge instance (Tesla T4 typically supports CUDA 11.x, 12.x).
# Check llama-cpp-python's documentation for recommended CUDA versions.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
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
    GCC_LIB_PATH="/usr/lib/gcc/$(gcc -dumpmachine)/$(gcc -dumpversion)" && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${GCC_LIB_PATH}:${LD_LIBRARY_PATH}" && \
    echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH" && \
    \
    # Set compiler and CMake flags for llama-cpp-python installation
    export CC=/usr/bin/gcc && \
    export CXX=/usr/bin/g++ && \
    export FORCE_CMAKE=1 && \
    # For g4dn (Tesla T4), CUDA compute capability is 7.5. Broader set could be "70;75;80;86;89"
    export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF" && \
    echo "CMAKE_ARGS: ${CMAKE_ARGS}" && \
    echo "FORCE_CMAKE: ${FORCE_CMAKE}" && \
    echo "CC: ${CC}" && \
    echo "CXX: ${CXX}" && \
    \
    # Install llama-cpp-python using uv pip, following latest GitHub issue success
    uv pip install --no-cache --force-reinstall "llama-cpp-python[server]==${LLAMA_CPP_PYTHON_VERSION}" && \
    \
    # Verify llama-cpp-python installation
    echo "Verifying llama-cpp-python installation..." && \
    python -m llama_cpp.server --help > /dev/null && \
    # The following python import test is a good check but might fail if runtime CUDA libs aren't perfectly set up yet for this check
    # python -c "from llama_cpp import llama_cpp; print(llama_cpp.ggml_cpu_has_avx())" && \
    echo "llama-cpp-python installation step completed."

# Ensure libcuda.so.1 is available in /usr/local/cuda/lib64, pointing to the stub.
# This is crucial for the linker to find it at runtime.
# This RUN layer is separate to ensure LD_LIBRARY_PATH from an earlier layer doesn't affect it if it were combined.
# However, the LD_LIBRARY_PATH set by ENV should persist for subsequent RUN layers.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/lib/gcc/x86_64-linux-gnu/11:${LD_LIBRARY_PATH}
RUN TARGET_CUDA_LIB_DIR="/usr/local/cuda/lib64" && \
    mkdir -p "$TARGET_CUDA_LIB_DIR" && \
    STUB_PATH_ABS_PRIMARY="$CUDA_TOOLKIT_ROOT_DIR/lib64/stubs/libcuda.so" && \
    STUB_PATH_ABS_SECONDARY="$CUDA_TOOLKIT_ROOT_DIR/lib64/libcuda.so" && \
    SYMLINK_TARGET="$TARGET_CUDA_LIB_DIR/libcuda.so.1" && \
    echo "Attempting to create symlink $SYMLINK_TARGET..." && \
    if [ -f "$STUB_PATH_ABS_PRIMARY" ]; then \
        ln -sf "$STUB_PATH_ABS_PRIMARY" "$SYMLINK_TARGET" && echo "Symlinked $STUB_PATH_ABS_PRIMARY to $SYMLINK_TARGET"; \
    elif [ -f "$STUB_PATH_ABS_SECONDARY" ]; then \
        ln -sf "$STUB_PATH_ABS_SECONDARY" "$SYMLINK_TARGET" && echo "Symlinked $STUB_PATH_ABS_SECONDARY to $SYMLINK_TARGET"; \
    else \
        echo "Error: CUDA stub libcuda.so not found in $STUB_PATH_ABS_PRIMARY or $STUB_PATH_ABS_SECONDARY. Cannot create $SYMLINK_TARGET." >&2; \
        exit 1; \
    fi && \
    echo "Verifying symlink $SYMLINK_TARGET:" && ls -l "$SYMLINK_TARGET" && \
    echo "Target of symlink $(readlink -f "$SYMLINK_TARGET") details:" && ls -l "$(readlink -f "$SYMLINK_TARGET")" && \
    echo "Contents of $TARGET_CUDA_LIB_DIR:" && ls -l "$TARGET_CUDA_LIB_DIR" && \
    echo "Contents of $CUDA_TOOLKIT_ROOT_DIR/lib64/stubs/ (if it exists):" && (ls -l "$CUDA_TOOLKIT_ROOT_DIR/lib64/stubs/" || echo "Directory not found or empty") && \
    # Also ensure libnvidia-ml.so.1 stub is linked if present (good practice)
    STUB_NVMKL_PATH_ABS="$CUDA_TOOLKIT_ROOT_DIR/lib64/stubs/libnvidia-ml.so" && \
    SYMLINK_NVMKL_TARGET="$TARGET_CUDA_LIB_DIR/libnvidia-ml.so.1" && \
    if [ -f "$STUB_NVMKL_PATH_ABS" ]; then \
        ln -sf "$STUB_NVMKL_PATH_ABS" "$SYMLINK_NVMKL_TARGET" && echo "Symlinked $STUB_NVMKL_PATH_ABS to $SYMLINK_NVMKL_TARGET"; \
    else \
        echo "Info: CUDA stub libnvidia-ml.so not found in $STUB_NVMKL_PATH_ABS. libnvidia-ml.so.1 link not created (this may be okay)."; \
    fi

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
