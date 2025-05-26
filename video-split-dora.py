import cv2
import numpy as np
from dora import Node
from PIL import Image

node = Node()

frames = {}
i = 0

for event in node:
    event_type = event["type"]
    if event_type == "INPUT":
        event_id = event["id"]

        if "image" in event_id:
            storage = event["value"]
            metadata = event["metadata"]
            encoding = metadata["encoding"]
            width = metadata["width"]
            height = metadata["height"]

            if (
                encoding == "bgr8"
                or encoding == "rgb8"
                or encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]
            ):
                channels = 3
                storage_type = np.uint8
            else:
                raise RuntimeError(f"Unsupported image encoding: {encoding}")

            if encoding == "bgr8":
                frame = (
                    storage.to_numpy()
                    .astype(storage_type)
                    .reshape((height, width, channels))
                )
                frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
            elif encoding == "rgb8":
                frame = (
                    storage.to_numpy()
                    .astype(storage_type)
                    .reshape((height, width, channels))
                )
            elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                storage = storage.to_numpy()
                frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
            else:
                raise RuntimeError(f"Unsupported image encoding: {encoding}")
            image = Image.fromarray(frame)

            # Save PIL Image
            image.save(f"images/image_{i}.jpg")
            i += 1
