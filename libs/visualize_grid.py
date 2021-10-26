from typing import List

import cv2
import numpy as np


def make_grid(images: List[List[np.ndarray]], pad: int = 3) -> np.ndarray:
    for k in range(len(images)):
        c, h, w = images[k][0].shape
        dtype = images[k][0].dtype
        row_image = images[k][0].transpose([1, 2, 0]) * 0.5 + 0.5
        for i in range(1, len(images[k])):
            add_image = cv2.hconcat(
                [
                    np.zeros([h, pad, c], dtype=dtype),
                    images[k][i].transpose([1, 2, 0]) * 0.5 + 0.5,
                ]
            )
            row_image = cv2.hconcat([row_image, add_image])

        if k == 0:
            grid_image = row_image
        else:
            h, w, c = row_image.shape
            add_image = cv2.vconcat([np.zeros([pad, w, c], dtype=dtype), row_image])
            grid_image = cv2.vconcat([grid_image, add_image])

    return grid_image


def make_grid_gray(images: List[List[np.ndarray]], pad: int = 3) -> np.ndarray:
    for k in range(len(images)):
        _, h, w = images[k][0].shape
        dtype = images[k][0].dtype
        row_image = images[k][0].transpose([1, 2, 0]) * 0.5 + 0.5
        for i in range(1, len(images[k])):
            add_image = cv2.hconcat(
                [
                    np.zeros([h, pad, 1], dtype=dtype),
                    images[k][i].transpose([1, 2, 0]) * 0.5 + 0.5,
                ]
            )
            row_image = cv2.hconcat([row_image, add_image])

        if k == 0:
            grid_image = row_image
        else:
            h, w = row_image.shape
            add_image = cv2.vconcat([np.zeros([pad, w], dtype=dtype), row_image])
            grid_image = cv2.vconcat([grid_image, add_image])

    return grid_image[:, :, np.newaxis]
