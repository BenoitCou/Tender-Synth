"""
Small helpers for image hashing, re-encoding, etc.
"""
import io
from typing import Optional

from PIL import Image

def convert_image_bytes(image_bytes: bytes, output_format: str = "PNG") -> Optional[io.BytesIO]:
    """
    Convert raw image_bytes to a Pillow-compatible ``BytesIO`` object in a
    standard raster format (default PNG). python-docx requires such an
    object when adding a picture to a document.

    Parameters
    ----------
    image_bytes : bytes
        The original binary payload of the image (any format readable by
        Pillow, e.g. JPEG, TIFF, etc.).
    output_format : str, default "PNG"
        Pillow format name to save the re-encoded image in.  PNG is a safe
        default because it is lossless and widely supported.

    Returns
    -------
    io.BytesIO | None
        A seek-reset ``BytesIO`` ready to be passed to
        ``doc.add_picture(stream, ...)``.  ``None`` is returned if Pillow
        fails to open or save the image.
    """

    try:
        # Load the original image from the raw byte payload
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Re-encode to the requested format
            output = io.BytesIO()
            img.save(output, format=output_format)
            output.seek(0)
            return output
        
    except Exception as e:
        # Pillow failed (corrupted image, unsupported format, etc.)
        print("Erreur lors de la conversion de l'image :", e)
        return None

