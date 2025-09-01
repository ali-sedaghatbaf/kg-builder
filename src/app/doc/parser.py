import logging
import os
from pathlib import Path

from llama_cloud_services import LlamaParse

from app.config import settings

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, mode="parse_page_without_llm"):
        api_key = (
            settings.LLAMA_CLOUD_API_KEY.get_secret_value()
            if settings.LLAMA_CLOUD_API_KEY
            else ""
        )
        self.parser = LlamaParse(parse_mode=mode, api_key=api_key)

    async def parse_async(self, file_name: str, page_count: int | None = None) -> str:
        """
        Parses the PDF file and returns the content as a string.
        """

        file_path = Path(__file__).parent.parent.parent / "uploads" / file_name

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Parse the contract text from the PDF
        if not file_path.suffix.lower() == ".pdf":
            raise ValueError("File is not a PDF")

        logger.info("Parsing PDF file: %s", file_path)

        results = await self.parser.aparse(str(file_path))
        if isinstance(results, list):
            results = results[0]
        text = ""

        markdown_docs = await results.aget_markdown_documents(split_by_page=True)

        for page in markdown_docs:
            text += page.text + "\n"

        return text
