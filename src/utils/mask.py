import base64
import zlib

import cv2
import numpy as np
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
