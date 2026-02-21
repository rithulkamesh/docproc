"""Serve the DocProc API with uvicorn. Loads config from docproc.yaml on startup."""

import os
import uvicorn


def main():
    from docproc.config import load_config
    load_config(os.getenv("DOCPROC_CONFIG"))
    from docproc.api.app import create_app
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
