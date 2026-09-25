"""Perspective billboard scene for an explicit, simplified closed-loop test.

The camera looks along world +x; world +y projects right and +z projects up.
The person is an unsegmented photographic patch on a camera-facing billboard.
Ground truth is returned separately and must never be supplied to detection or
the controller. This renderer is not a physical camera or aerodynamic simulator.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np


SOURCE_SEQUENCE = 315
SOURCE_BBOX = (1128, 431, 1307, 944)


def extract_person_patch(video_path, sequence=SOURCE_SEQUENCE, bbox=SOURCE_BBOX, margin=15):
    """Sequentially decode an exact source index and copy its person rectangle.

    Returns RGB patch plus crop/source metadata, with no downloaded media.
    Source licensing is retained by the caller in the experiment's provenance.
    """
    import cv2
    if not isinstance(sequence, Integral) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("sequence must be a nonnegative integer")
    if not isinstance(margin, Integral) or isinstance(margin, bool) or margin < 0:
        raise ValueError("margin must be a nonnegative integer")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("source video could not be opened")
        for _ in range(int(sequence)+1):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError("source ended before requested sequence")
        h, w = bgr.shape[:2]
        if (not isinstance(bbox, (tuple, list)) or len(bbox) != 4
                or not all(isinstance(v, Real) and np.isfinite(v) for v in bbox)
                or not 0 <= bbox[0] < bbox[2] <= w or not 0 <= bbox[1] < bbox[3] <= h):
            raise ValueError("bbox must be an on-image rectangle")
        x0, y0 = max(0, int(np.floor(bbox[0]))-margin), max(0, int(np.floor(bbox[1]))-margin)
        x1, y1 = min(w, int(np.ceil(bbox[2]))+margin), min(h, int(np.ceil(bbox[3]))+margin)
        patch = np.ascontiguousarray(bgr[y0:y1, x0:x1, ::-1])
        return patch, {"source_sequence": int(sequence), "crop_xyxy": [x0, y0, x1, y1],
                       "person_bbox_source_xyxy": list(bbox), "source_image_hw": [h, w],
                       "patch_image_hw": list(patch.shape[:2]),
                       "representation": "unsegmented source photo crop on a camera-facing billboard"}
    finally:
        capture.release()


def render_scene(person_patch, vehicle_position, target_position, *, focal_px=360.,
                 physical_patch_height_m=1.9, image_size=391, hidden=False):
    """Render one RGB image and a separate projection truth dictionary.

    Positions are (world x, y, z), with target_position at the patch's centre.
    Moving the camera changes the next projection, closing the visual feedback
    loop when the caller integrates its commands. No depth is encoded as an API
    detection: ``relative_depth_m`` exists only in returned evaluation truth.
    """
    import cv2
    if (not isinstance(person_patch, np.ndarray) or person_patch.dtype != np.uint8
            or person_patch.ndim != 3 or person_patch.shape[2] != 3 or min(person_patch.shape[:2]) < 2):
        raise ValueError("person_patch must be uint8 RGB HWC")
    vehicle, target = np.asarray(vehicle_position, dtype=float), np.asarray(target_position, dtype=float)
    if vehicle.shape != (3,) or target.shape != (3,) or not np.isfinite(vehicle).all() or not np.isfinite(target).all():
        raise ValueError("positions must be finite world (x,y,z) triples")
    if (isinstance(focal_px, bool) or not isinstance(focal_px, Real) or not np.isfinite(focal_px) or focal_px <= 0
            or isinstance(physical_patch_height_m, bool) or not isinstance(physical_patch_height_m, Real)
            or not np.isfinite(physical_patch_height_m) or physical_patch_height_m <= 0):
        raise ValueError("focal length and physical patch height must be positive and finite")
    if not isinstance(image_size, Integral) or isinstance(image_size, bool) or not 16 <= image_size <= 2048:
        raise ValueError("image_size must be an integer in [16,2048]")
    if type(hidden) is not bool:
        raise ValueError("hidden must be boolean")
    image_size = int(image_size)
    # A deliberately plain background gives a repeatable engineering fixture;
    # the scene is not a photorealistic person-detection benchmark.
    yy, xx = np.indices((image_size, image_size))
    base = np.clip(174 - 28*yy/image_size, 0, 255).astype(np.uint8)
    rgb = np.stack((base, base+np.uint8(6), base+np.uint8(9)), axis=-1)
    rgb[yy > image_size*.73] = (137, 143, 147)
    relative = target-vehicle
    truth = {"visible": False, "hidden": hidden, "bbox_xyxy": None,
             "unclipped_bbox_xyxy": None, "relative_depth_m": float(relative[0]),
             "relative_position_m": relative.tolist(), "image_hw": [image_size, image_size],
             "world_axes": "+x camera forward, +y image right, +z image up",
             "rendering": "camera-facing unsegmented photographic billboard"}
    if hidden or relative[0] <= .05:
        return rgb, truth
    projected_h = float(focal_px*physical_patch_height_m/relative[0])
    projected_w = projected_h*person_patch.shape[1]/person_patch.shape[0]
    cx = image_size/2 + focal_px*relative[1]/relative[0]
    cy = image_size/2 - focal_px*relative[2]/relative[0]
    left, top, right, bottom = cx-projected_w/2, cy-projected_h/2, cx+projected_w/2, cy+projected_h/2
    truth["unclipped_bbox_xyxy"] = [float(left), float(top), float(right), float(bottom)]
    x0, y0, x1, y1 = max(0.,left), max(0.,top), min(float(image_size),right), min(float(image_size),bottom)
    if x0 >= x1 or y0 >= y1:
        return rgb, truth
    # Warp directly into the bounded image instead of allocating a huge scaled
    # crop when the camera approaches the near plane.
    transform = np.array([[projected_w/person_patch.shape[1], 0., left],
                          [0., projected_h/person_patch.shape[0], top]], dtype=float)
    warped = cv2.warpAffine(person_patch, transform, (image_size, image_size),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inside = ((xx+.5 >= left) & (xx+.5 < right) & (yy+.5 >= top) & (yy+.5 < bottom))
    rgb[inside] = warped[inside]
    truth.update(visible=bool(inside.any()), bbox_xyxy=[x0, y0, x1, y1])
    return np.ascontiguousarray(rgb), truth


# The shorter name makes a renderer callback convenient without concealing the
# explicit scene semantics from callers that use the descriptive name.
render = render_scene
