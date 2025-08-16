"""Start the MedMDT API server.

Usage:
    python scripts/server.py [--host HOST] [--port PORT]
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="MedMDT API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    uvicorn.run(
        "medmdt.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
