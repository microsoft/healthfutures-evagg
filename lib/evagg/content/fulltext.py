from dataclasses import dataclass
from typing import Generator, List, Optional
from xml.etree.ElementTree import Element

SectionFilter = Optional[List[str]]
# Sections that are often not considered "content" sections.
NON_CONTENT_SECTION_TYPES = ["ABBR", "ACK_FUND", "AUTH_CONT", "COMP_INT", "REF"]


@dataclass(frozen=True)
class TextSection:
    section_type: str  # Type of section (e.g., ABSTRACT, METHODS, RESULTS, TABLE)
    text_type: str  # Type of section text (e.g., "fig_caption", "paragraph", "title_2")
    offset: int  # Offset within the document
    text: str  # Raw text from the passage
    id: Optional[str] = None  # Available on TABLEs and FIGs


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
        id = passage.findtext("infon[@key='id']")
        # Eliminate linebreaks and strip leading/trailing whitespace from each line.
        text = " ".join(s.strip() for s in (passage.findtext("text") or "").split("\n"))
        yield TextSection(section_type, text_type, int(offset), text, id)


def get_section_texts(
    doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None
) -> Generator[str, None, None]:
    """Filter to the given sections in the XML full-text document and return the texts."""

    def _include_section(section: TextSection) -> bool:
        if include and section.section_type not in include:
            return False
        if exclude and section.section_type in exclude:
            return False
        return True

    for section in get_fulltext_sections(doc):
        if _include_section(section):
            yield section.text


def get_fulltext(doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None) -> str:
    """Extract and join the text with newlines from the given sections in the XML full-text document."""
    return "\n".join(get_section_texts(doc, include, exclude))
