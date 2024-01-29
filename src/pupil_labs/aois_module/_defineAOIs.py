import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rich.logging import RichHandler  # For logging
from starlette.responses import FileResponse

from pupil_labs.aois_module._AOIs import AOI_Generator
from pupil_labs.aois_module._to_cloud import post_aoi

logging.getLogger("defineAOIs")
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)
static_folder = Path(__file__).parent / "static"


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
app.mount("/static", StaticFiles(directory=static_folder), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SegmentRequest(BaseModel):
    image: str
    text: str


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_folder, "index.html"))


@app.post("/segment")
async def segment(request: SegmentRequest):
    image = app.AOI.decode_img(request.image)
    image = app.AOI.scale_img(image)

    pred_names, boxes = app.AOI.predict_dino(image, request.text)
    aois = app.AOI.predict_sam(image, pred_names, boxes)

    segments_list = aois.to_dict(orient="records")
    formatted_segments = [
        {"label": segment["label"], "mask": segment["mask"]}
        for segment in segments_list
    ]
    logging.info(formatted_segments)
    return JSONResponse(content=formatted_segments)


@app.post("/to_cloud")
async def to_cloud(
    image: UploadFile = File(), search: str = "", token: str = "", url: str = ""
):
    """This function processes an uploaded image and sends data to cloud."""

    file_path = os.path.join(Path(__file__).parent, "image.jpeg")
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    import cv2

    image = cv2.imread(file_path)
    image = app.AOI.scale_img(image)
    pred_names, boxes = app.AOI.predict_dino(image, search)
    aois = app.AOI.predict_sam(image, pred_names, boxes)
    for index, aoi in aois.iterrows():
        post_aoi(aoi['mask'], aoi['label'], [], index, url, token)

    return {"message": "Processed and data sent to cloud"}


def main():
    nest_asyncio.apply()
    uvicorn.run(
        app,
        port=8002,
    )


if __name__ == "__main__":
    main()
