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

    def _include_section(section_type: str) -> bool:
        """Check if the section should be included based on the filters. Exclude takes precedence over include."""
        if include and section_type not in include:
            return False
        if exclude and section_type in exclude:
            return False
        return True

    doc_id = doc.findtext("id") or "unknown"
    # Generate all "passage" elements.
    for passage in doc.iterfind("./passage"):
        if not (section_type := passage.findtext("infon[@key='section_type']")):
            raise ValueError(f"Missing 'section_type' infon element in passage for document {doc_id}")
        if not _include_section(section_type):
            continue
        if not (text_type := passage.findtext("infon[@key='type']")):
            raise ValueError(f"Missing 'type' infon element in passage for document {doc_id}")
        if not (offset := passage.findtext("offset")):
            raise ValueError(f"Missing 'offset' infon element in {section_type} passage for document {doc_id}")
        if not (id := passage.findtext("infon[@key='id']")):
            id = "none"
        # Eliminate linebreaks and strip leading/trailing whitespace from each line.
        text = " ".join(s.strip() for s in (passage.findtext("text") or "").split("\n"))
        yield TextSection(section_type, text_type, int(offset), text, id)


def get_section_texts(
    doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None
) -> Generator[str, None, None]:
    """Filter to the given sections in the XML full-text document and return the texts."""
    for section in get_sections(doc, include, exclude):
        yield section.text


def get_fulltext(doc: Optional[Element], include: SectionFilter = None, exclude: SectionFilter = None) -> str:
    """Extract and join the text with newlines from the given sections in the XML full-text document."""
    return "\n".join(get_section_texts(doc, include, exclude))
