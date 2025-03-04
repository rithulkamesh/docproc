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
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${
      pkgs.lib.makeLibraryPath [
        pkgs.mesa
        pkgs.libGL
        pkgs.xorg.libX11
        pkgs.glib
      ]
    }:$LD_LIBRARY_PATH

    # Initialize and activate a Python virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      python -m venv .venv
      echo "Virtual environment created."
    fi
    source .venv/bin/activate
    echo "Virtual environment activated."

    # Automatically run 'uv sync'
    uv sync
  '';
}
