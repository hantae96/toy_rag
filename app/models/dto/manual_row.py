from pydantic import BaseModel


class ManualRow(BaseModel):
    id: int | str
    post_id: int | str
    category_id: int | str
    category: str
    version: int | str
    title: str
    content: str
    writed_by : str
