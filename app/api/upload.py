"""Reusable streaming read with encoded-image byte limit."""

from fastapi import UploadFile

from app.image_service import MAX_FILE_SIZE, ImageTooLargeError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
UPLOAD_CHUNK_SIZE = 1024 * 1024


async def read_upload_limited(image: UploadFile) -> bytes:
    contents = bytearray()
    while True:
        chunk = await image.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        if len(contents) + len(chunk) > MAX_FILE_SIZE:
            raise ImageTooLargeError("Image size must not exceed 10 MiB.")
        contents.extend(chunk)
    return bytes(contents)
