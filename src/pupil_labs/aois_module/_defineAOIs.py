import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

import cv2
import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pupil_labs.aois_module._AOIs import AOI_Generator
from pupil_labs.aois_module._to_cloud import delete_aois, list_aois, post_aoi
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from starlette.responses import FileResponse

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
    search: str


class Segment(BaseModel):
    label: str
    mask: str


class SegmentResponse(BaseModel):
    segments: List[Segment]
    image: str


class CloudDetails(BaseModel):
    token: str = Field(default="")
    url: str = Field(default="")


class CloudPayload(CloudDetails):
    formatted_segments: List[Segment]


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_folder, "index.html"))


@app.post("/scale")
async def scale(image: UploadFile = File(...)) -> Any:
    """
    Scales the uploaded image using predefined settings.

    Args:
        image (UploadFile): The image file to be scaled.

    Returns:
        FileResponse: The scaled image as a file response.
    """
    try:
        temp_file_path = Path(__file__).parent / "temp_image.jpg"

        with temp_file_path.open("wb") as buffer:
            buffer.write(await image.read())

        img = cv2.imread(str(temp_file_path))
        scaled_img = app.AOI.scale_img(img)
        scaled_img_enc = app.AOI.encode_img(scaled_img)
        logging.info(f"New size:{scaled_img.shape}")
        return JSONResponse(
            content={
                "height": int(scaled_img.shape[0]),
                "width": int(scaled_img.shape[1]),
                "image": scaled_img_enc,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scaling the image: {e}")
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()


@app.post("/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest) -> SegmentResponse:
    """
    Segments the uploaded image based on the provided text.

    Args:
        image (UploadFile): The image file to segment.
        text (str): The text describing what to segment in the image.

    Returns:
        SegmentResponse: The segmentation result including labels and masks, and the processed image.
    """
    try:
        img = app.AOI.decode_img(request.image)
        pred_names, boxes = app.AOI.predict_dino(img, request.search)
        aois, c_masks = app.AOI.predict_sam(img, pred_names, boxes)

        segments_list = aois.to_dict(orient="records")
        formatted_segments = [
            {"label": segment["label"], "mask": segment["mask"]}
            for segment in segments_list
        ]

        logging.info(formatted_segments)

        img = app.AOI.paint_image(img, c_masks)
        encoded_image = app.AOI.encode_img(img)

        return JSONResponse(
            content={"segments": formatted_segments, "image": encoded_image}
        )
    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/to_cloud")
async def to_cloud(payload: CloudPayload):
    """
    Sends formatted segment data to the cloud.

    Args:
        payload (CloudPayload): A payload containing formatted segments, a token, and a URL.

    Returns:
        dict: A message indicating the result of the operation.
    """
    try:
        for index, aoi in enumerate(payload.formatted_segments):
            post_aoi(aoi.mask, aoi.label, [], index, payload.url, payload.token)
        return {"message": "Processed and data sent to cloud"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to send data to cloud: {str(e)}"
        )


@app.post("/delete")
async def delete(payload: CloudDetails):
    """
    Sends formatted segment data to the cloud.

    Args:
        payload (CloudPayload): A payload containing formatted segments, a token, and a URL.

    Returns:
        dict: A message indicating the result of the operation.
    """
    try:
        aoi_ids = list_aois(url=payload.url, api_key=payload.token)
        logging.info(aoi_ids)
        delete_aois(url=payload.url, api_key=payload.token, aoi_ids=aoi_ids)
        return {"message": "Processed and data sent to cloud"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete data from cloud: {str(e)}"
        )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(static_folder, "favicon.ico"))


def main():
    nest_asyncio.apply()
    uvicorn.run(app, port=8002, log_level="info")


if __name__ == "__main__":
    main()
