import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import MODEL_DOWNLOAD_ROOT


def _load_chat_html() -> str:
    path = os.path.join(os.path.dirname(__file__), "templates", "chat.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_UI_TEST_PAGE_HTML = _load_chat_html()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        os.makedirs(os.path.join(MODEL_DOWNLOAD_ROOT, "models"), exist_ok=True)
    except OSError:
        pass
    yield


app = FastAPI(
    title="OpenAI-compatible LiteRT-LM",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/v1")

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
def root_redirect_ui():
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui_test_page():
    return HTMLResponse(content=_UI_TEST_PAGE_HTML, media_type="text/html; charset=utf-8")
