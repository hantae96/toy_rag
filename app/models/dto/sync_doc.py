from dataclasses import dataclass


@dataclass(slots=True)
class Manual:
    id: str
    post_id: str
    version: str
    title: str
    doc_url: str
    text: str
    writed_by: str
