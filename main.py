import uvicorn

from src.app.config import get_settings

settings = get_settings()


def main():
    uvicorn.run(
        "src.app.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
