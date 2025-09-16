import logging
import os
from pathlib import Path

from llama_cloud_services import LlamaParse

from ..config import get_settings

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, mode="parse_page_without_llm"):
        settings = get_settings()
        api_key = (
            settings.LLAMA_CLOUD_API_KEY.get_secret_value()
            if settings.LLAMA_CLOUD_API_KEY
            else ""
        )
        self.parser = LlamaParse(parse_mode=mode, api_key=api_key)

    async def aparse(self, file_name: str, page_count: int | None = None) -> str:
        """
        Parses the PDF file and returns the content as a string.
        """
        file_path = Path(__file__).parent.parent.parent.parent / "uploads" / file_name

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Parse the contract text from the PDF
        if not file_path.suffix.lower() == ".pdf":
            raise ValueError("File is not a PDF")

        logger.info("Parsing PDF file: %s", file_path)

        results = await self.parser.aparse(str(file_path))
        if isinstance(results, list):
            results = results[0]

        # Prefer page-wise markdown aggregation
        text = ""
        try:
            markdown_docs = await results.aget_markdown_documents(split_by_page=True)
        except Exception:
            markdown_docs = None

        if markdown_docs:
            if page_count:
                markdown_docs = markdown_docs[:page_count]
            for page in markdown_docs:
                page_text = None
                # Support multiple possible shapes
                if hasattr(page, "text"):
                    page_text = getattr(page, "text")
                elif hasattr(page, "md"):
                    page_text = getattr(page, "md")
                elif isinstance(page, dict):
                    page_text = (
                        page.get("text") or page.get("markdown") or page.get("md")
                    )
                if page_text:
                    text += str(page_text) + "\n"

        # Fallbacks if page-wise markdown is empty
        if not text.strip():
            try:
                md = await results.aget_markdown()
                if md and str(md).strip():
                    return str(md)
            except Exception:
                pass
            try:
                plain = await results.aget_text()
                if plain and str(plain).strip():
                    return str(plain)
            except Exception:
                pass

        return text
