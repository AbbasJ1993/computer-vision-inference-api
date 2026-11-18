"""Validation and safe image decoding for the CV inference API.

Validation and decoding are intentionally separate so inference can verify the
encoded image once and fully decode it only once. ``inspect_image`` keeps the
existing strict metadata-only endpoint behavior (verify + complete decode).
"""
from io import BytesIO

from PIL import Image, UnidentifiedImageError


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 16_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImageError(Exception):
    """Raised when the uploaded file is not a valid supported image."""


class ImageTooLargeError(Exception):
    """Raised when the uploaded image exceeds the size limit."""


class ImageDimensionsTooLargeError(Exception):
    """Raised when image dimensions exceed the allowed limits."""


def validate_image_structure(image_bytes: bytes) -> dict:
    """Validate limits, format and encoded structure; return image metadata.

    This performs ``verify()`` but NOT a full pixel decode. Callers must
    subsequently decode all pixels before trusting the image. The public
    ``inspect_image`` and the inference service both enforce this rule.
    """
    if not image_bytes:
        raise InvalidImageError("The uploaded file is empty.")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise ImageTooLargeError("Image size must not exceed 10 MiB.")

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            image_format = img.format
            if image_format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    "Only JPEG, PNG, and WEBP images are supported."
                )

            width, height = img.size
            if (
                width > MAX_IMAGE_WIDTH
                or height > MAX_IMAGE_HEIGHT
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ImageDimensionsTooLargeError(
                    "Image dimensions exceed the allowed limits."
                )
            img.verify()

    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        ) from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "The image is corrupted or invalid."
        ) from exc

    return {
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(image_bytes),
    }


def _decode_pixels(image_bytes: bytes, *, rgb: bool) -> Image.Image | None:
    """Decode image fully, translating Pillow errors to our API exception."""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            if rgb:
                # convert() loads pixels and returns an independent RGB image.
                # It is safe to use after the encoded stream is closed.
                result = img.convert("RGB")
                result.load()
                return result
            img.load()
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        ) from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "The image is corrupted or invalid."
        ) from exc
    return None


def decode_rgb_image(image_bytes: bytes) -> Image.Image:
    """Full RGB pixel decode; call ONLY after ``validate_image_structure``."""
    result = _decode_pixels(image_bytes, rgb=True)
    assert result is not None
    return result


def inspect_image(image_bytes: bytes) -> dict:
    """Strict metadata-only inspection, preserving prior endpoint behavior.

    A full decode still occurs, rejecting truncated images, even though the
    caller does not need RGB pixels. This endpoint is intentionally unchanged.
    """
    metadata = validate_image_structure(image_bytes)
    _decode_pixels(image_bytes, rgb=False)
    return metadata
