"""Server entrypoint module expected by OpenEnv multi-mode validators."""

import os
import uvicorn

from app import app


def main() -> None:
    """Run the FastAPI app."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
