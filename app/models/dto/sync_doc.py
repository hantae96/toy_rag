from dataclasses import dataclass, field


@dataclass(slots=True)
class Manual:
    id: str
    post_id: str
    version: str
    title: str
    doc_url: str
    text: str
    image_urls: list[str] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)
    content_sequence: list[dict[str, str]] = field(default_factory=list)
