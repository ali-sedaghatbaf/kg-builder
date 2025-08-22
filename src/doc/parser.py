import logging
import os
from pathlib import Path

from llama_cloud_services import LlamaParse

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, mode="parse_page_without_llm"):
        self.parser = LlamaParse(
            parse_mode=mode, api_key=os.getenv("LLAMA_CLOUD_API_KEY")
        )

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
        result = await self.parser.aparse(file_path)
        if page_count is None:
            page_count = len(result.pages)
        text = " ".join([el.text for el in result.pages[:page_count]])
        return text
