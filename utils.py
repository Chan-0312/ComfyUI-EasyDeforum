from typing import List, Union
import numpy as np
import torch
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from math import sqrt, ceil
from typing import cast


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def image_2dtransform(
        image,
        x,
        y,
        zoom,
        angle,
        shear=0,
        border_handling="edge",
    ):
        x = int(x)
        y = int(y)
        angle = int(angle)

        if image.size(0) == 0:
            return (torch.zeros(0),)
        frames_count, frame_height, frame_width, frame_channel_count = image.size()

        new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)

        # - Calculate diagonal of the original image
        diagonal = sqrt(frame_width**2 + frame_height**2)
        max_padding = ceil(diagonal * zoom - min(frame_width, frame_height))
        # Calculate padding for zoom
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [max(0, pw + x), max(0, ph + y), max(0, pw - x), max(0, ph - y)]

        img = tensor2pil(image)[0]

        img = TF.pad(
            img,  # transformed_frame,
            padding=padding,
            padding_mode=border_handling,
        )

        img = cast(
            Image.Image,
            TF.affine(img, angle=angle, scale=zoom, translate=[x, y], shear=shear),
        )

        left = abs(padding[0])
        upper = abs(padding[1])
        right = img.width - abs(padding[2])
        bottom = img.height - abs(padding[3])

        img = img.crop((left, upper, right, bottom))

        return pil2tensor(img)