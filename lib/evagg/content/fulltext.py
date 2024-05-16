from typing import Generator, List, Optional
from xml.etree.ElementTree import Element

from .interfaces import TextSection

SectionFilter = Optional[List[str]]


def get_sections(
    doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None
) -> Generator[TextSection, None, None]:
    """Build a list of TextSection objects from the XML full-text document."""
    if doc is None:
        return

    def _include_section(section: TextSection) -> bool:
        if include and section.section_type not in include:
            return False
        if exclude and section.section_type in exclude:
            return False
        return True

    doc_id = doc.findtext("id") or "unknown"
    # Generate all "passage" elements.
    for passage in doc.iterfind("./passage"):
        if not (section_type := passage.findtext("infon[@key='section_type']")):
            raise ValueError(f"Missing 'section_type' infon element in passage for document {doc_id}")
        if not (text_type := passage.findtext("infon[@key='type']")):
            raise ValueError(f"Missing 'type' infon element in passage for document {doc_id}")
        if not (offset := passage.findtext("offset")):
            raise ValueError(f"Missing 'offset' infon element in {section_type} passage for document {doc_id}")
        text = " ".join(s.strip() for s in (passage.findtext("text") or "").split("\n"))
        section = TextSection(section_type, text_type, int(offset), text)
        if _include_section(section):
            # Eliminate linebreaks and strip leading/trailing whitespace from each line.
            yield section


def get_section_texts(
    doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None
) -> Generator[str, None, None]:
    """Filter to the given sections in the XML full-text document and return the texts."""
    for section in get_sections(doc, include, exclude):
        yield section.text


def get_all_sections(doc: Optional[Element]) -> Generator[TextSection, None, None]:
    """Extract all sections from the XML full-text document."""
    return get_sections(doc)


def get_fulltext(doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None) -> str:
    """Extract and join the text with newlines from the given sections in the XML full-text document."""
    return "\n".join(get_section_texts(doc, include, exclude))
