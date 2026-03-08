import logging
from pathlib import Path
from pypdf import PdfReader
import docx

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {"txt", "md", "py", "json", "csv"}

class DocumentLoader:

    def load_file(self, file_path: str) -> str:
        path = Path(file_path)
        extension = path.suffix.lstrip(".").lower()

        loaders = {
            "pdf":  self._load_pdf,
            "docx": self._load_docx,
        }

        if extension in loaders:
            return loaders[extension](path)

        if extension in _TEXT_EXTENSIONS:
            return self._load_text(path)

        logger.warning("Unsupported file type '%s' — skipping %s", extension, path.name)
        return ""

    def _load_pdf(self, path: Path) -> str:
        pages: list[str] = []
        try:
            reader = PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, start=1):
                content = page.extract_text()
                if content:
                    pages.append(content)
                else:
                    logger.debug("Page %d of '%s' yielded no text.", page_num, path.name)
        except Exception:
            logger.exception("Failed to read PDF '%s'.", path.name)
        return "\n".join(pages)

    def _load_docx(self, path: Path) -> str:
        paragraphs: list[str] = []
        try:
            document = docx.Document(str(path))
            paragraphs = [
                para.text
                for para in document.paragraphs
                if para.text.strip()
            ]
        except Exception:
            logger.exception("Failed to read DOCX '%s'.", path.name)
        return "\n".join(paragraphs)

    def _load_text(self, path: Path) -> str:
        for encoding in ("utf-8", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception:
                logger.exception("Failed to read text file '%s'.", path.name)
                return ""
        logger.error("Could not decode '%s' with any supported encoding.", path.name)
        return ""