from dataclasses import dataclass
from typing import Generator, List, Optional
from xml.etree.ElementTree import Element

SectionFilter = Optional[List[str]]


@dataclass(frozen=True)
class TextSection:
    section_type: str
    text_type: str
    offset: int
    text: str


def get_fulltext_sections(doc: Optional[Element]) -> Generator[TextSection, None, None]:
    """Build a list of TextSection objects from the XML full-text document."""
    if doc is None:
        return
    doc_id = doc.findtext("id") or "unknown"
    # Generate all "passage" elements.
    for passage in doc.iterfind("./passage"):
        if not (section_type := passage.findtext("infon[@key='section_type']")):
            raise ValueError(f"Missing 'section_type' infon element in passage for document {doc_id}")
        if not (text_type := passage.findtext("infon[@key='type']")):
            raise ValueError(f"Missing 'type' infon element in passage for document {doc_id}")
        if not (offset := passage.findtext("offset")):
            raise ValueError(f"Missing 'offset' infon element in {section_type} passage for document {doc_id}")
        # Eliminate linebreaks and strip leading/trailing whitespace from each line.
        text = " ".join(s.strip() for s in (passage.findtext("text") or "").split("\n"))
        yield TextSection(section_type, text_type, int(offset), text)


def get_section_text(doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None) -> str:
    """Extract and join the text with newlines from the given elements in the XML full-text document."""

    def _include_section(section: TextSection) -> bool:
        if include and section.section_type not in include:
            return False
        if exclude and section.section_type in exclude:
            return False
        return True

    texts = [section.text for section in get_fulltext_sections(doc) if _include_section(section)]
    return "\n".join(texts)
