import logging
from contextlib import asynccontextmanager
from pathlib import Path

import nest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rich.logging import RichHandler  # For logging

from pupil_labs.aois_module._AOIs import AOI_Generator

# For parsing the arguments

logging.getLogger("defineAOIs")
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)

assets_folder = Path(__file__).parent / "assets"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(
        "[white bold on #0d122a]◎ Define AIOs using SAM by Pupil Labs[/]",
        extra={"markup": True},
    )
    logging.info(
        "[white bold on #4267B2]∞ Powered by Meta AI open source libraries[/]",
        extra={"markup": True},
    )
    app.AOI = AOI_Generator()
    yield
    logging.info("Sam went to sleep!")


app = FastAPI(title="GSAM AOI", version="0.0.1", lifespan=lifespan)


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
class SegmentRequest(BaseModel):
    image: str
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/segment")
async def segment(request: SegmentRequest):
    image = app.AOI.decode_img(request.image)
    image = app.AOI.scale_img(image)

    pred_names, boxes = app.AOI.predict_dino(image, request.text)
    aois = app.AOI.predict_sam(image, pred_names, boxes)

    segments_list = aois.to_dict(orient="records")
    formatted_segments = (
        {"label": segment["label"], "mask": segment["mask"]}
        for segment in segments_list
    )
    return JSONResponse(content={formatted_segments})


def main():
    nest_asyncio.apply()
    uvicorn.run(
        app,
        port=8002,
    )


if __name__ == "__main__":
    main()
