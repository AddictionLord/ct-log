import base64
import io
import zlib

import cv2
import numpy as np
from PIL import Image
import torch


def base64_to_mask(string: str) -> torch.Tensor:
    """Converts a base64 encoded string to a boolean mask tensor.

    Args:
        string: A string containing base64 encoded mask data.

    Returns:
        torch.Tensor: [H, W] dtype: bool, True indicates the presence of a pixel in the mask.
    """
    z = zlib.decompress(base64.b64decode(string))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)

    return torch.from_numpy(mask)


def mask_to_base64(mask: torch.Tensor) -> str:
    """Converts a boolean mask tensor to a base64 encoded string.

    Args:
        mask: A boolean tensor of shape [H, W] representing the mask.

    Returns:
        str: Base64 encoded string containing compressed PNG mask data.
    """
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])

    bytes_io = io.BytesIO()
    bytes_data = bytes_io.getvalue()

    img_pil.save(bytes_io, format="PNG", transparency=0, optimize=0)

    return base64.b64encode(zlib.compress(bytes_data)).decode("utf-8")
