from dataclasses import dataclass


@dataclass(slots=True)
class PreprocessDoc:
    doc_id: str
    preprcess_text: str
