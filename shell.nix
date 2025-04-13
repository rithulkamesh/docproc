{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  name = "docproc-shell";
  buildInputs = [
    pkgs.python310
    pkgs.mesa
    pkgs.libuv
    pkgs.libGL
    pkgs.xorg.libX11
    pkgs.glib
    pkgs.tesseract
    # Add CUDA-related packages
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.linuxPackages.nvidia_x11
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${
      pkgs.lib.makeLibraryPath [
        pkgs.mesa
        pkgs.libGL
        pkgs.xorg.libX11
        pkgs.glib
        pkgs.cudaPackages.cudatoolkit
        pkgs.cudaPackages.cudnn
        pkgs.linuxPackages.nvidia_x11
      ]
    }:$LD_LIBRARY_PATH

    # Set CUDA environment variables
    export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
    export CUDNN_HOME=${pkgs.cudaPackages.cudnn}
    export NVIDIA_DRIVER_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Initialize and activate a Python virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      python -m venv .venv
      echo "Virtual environment created."
    fi
    source .venv/bin/activate
    echo "Virtual environment activated."
    alias vi="nvim"

    # Automatically run 'uv sync'
    uv sync
  '';
}
