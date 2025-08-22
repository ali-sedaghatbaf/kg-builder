import os
from pathlib import Path

import uvicorn


async def main():
    if Path.exists(Path(__file__).parent / ".env"):
        from dotenv import load_dotenv

        load_dotenv()

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )


if __name__ == "__main__":
    main()
