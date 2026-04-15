from pydantic import BaseModel


class ModelDownloadRequest(BaseModel):
    url: str
    path: str


class ModelLoadRequest(BaseModel):
    path: str
    model_id: str | None = None
