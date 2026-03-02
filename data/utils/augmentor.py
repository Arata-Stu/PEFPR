import torch
import cv2
import numpy as np
import numba
from typing import List

# ==========================================
# Numba JIT functions for fast event processing
# ==========================================
@numba.njit
def _add_event(x, y, xlim, ylim, p, i, count, pos, mask, threshold=1.0):
    weight = 1.0 if p > 0 else -1.0
    
    count[ylim, xlim] += float(weight * (1 - abs(x - xlim)) * (1 - abs(y - ylim)))
    pol = 1 if count[ylim, xlim] > 0 else -1

    if pol * count[ylim, xlim] > threshold:
        count[ylim, xlim] -= pol * threshold
        mask[i] = True
        pos[i, 0] = xlim
        pos[i, 1] = ylim

@numba.njit
def _subsample(pos: np.ndarray, polarity: np.ndarray, mask: np.ndarray, count: np.ndarray, threshold=1.0):
    for i in range(len(pos)):
        x, y = pos[i]
        x0, x1 = int(x), int(x+1)
        y0, y1 = int(y), int(y+1)

        _add_event(x, y, x0, y0, polarity[i], i, count, pos, mask, threshold)
        _add_event(x, y, x1, y0, polarity[i], i, count, pos, mask, threshold)
        _add_event(x, y, x0, y1, polarity[i], i, count, pos, mask, threshold)
        _add_event(x, y, x1, y1, polarity[i], i, count, pos, mask, threshold)

# ==========================================
# Helper functions (NumPy/Dict basis)
# ==========================================
def _crop_events(events, left, right):
    valid = (events['x'] >= left[0]) & (events['x'] < right[0]) & \
            (events['y'] >= left[1]) & (events['y'] < right[1])

    return {k: v[valid] for k, v in events.items()}

def _crop_image(image, left, right):
    # image shape is [H, W, C]
    xmin, ymin = int(left[0]), int(left[1])
    xmax, ymax = int(right[0]), int(right[1])
    image[:ymin, :] = 0
    image[ymax:, :] = 0
    image[:, :xmin] = 0
    image[:, xmax:] = 0
    return image

def _resize_image(image, height, width, bg=None):
    # image shape is [H, W, C]
    new_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    px = (new_image.shape[1] - image.shape[1]) // 2
    py = (new_image.shape[0] - image.shape[0]) // 2

    if px >= 0:
        bg = new_image[py:py+image.shape[0], px:px+image.shape[1]]
    else:
        assert bg is not None
        bg[-py:-py+new_image.shape[0], -px:-px+new_image.shape[1]] = new_image

    return bg

def _crop_bbox(tracks, left, right):
    # tracks is assumed to be a numpy structured array with 'x', 'y', 'w', 'h'
    x1 = tracks['x']
    y1 = tracks['y']
    x2 = x1 + tracks['w']
    y2 = y1 + tracks['h']

    x1 = np.clip(x1, left[0], right[0])
    y1 = np.clip(y1, left[1], right[1])
    x2 = np.clip(x2, left[0], right[0])
    y2 = np.clip(y2, left[1], right[1])

    tracks['x'] = x1
    tracks['y'] = y1
    tracks['w'] = x2 - x1
    tracks['h'] = y2 - y1
    return tracks

def _scale_and_clip(x, scale):
    return int(np.clip(x * scale, 0, scale - 1))

# ==========================================
# Transforms
# ==========================================
class Compose:
    """torchvision.transforms.Compose の代わりとなるシンプルなクラス"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class RandomHFlip:
    def __init__(self, p: float):
        self.p = p
        self.width = None

    def init(self, height, width):
        self.width = width

    def __call__(self, data: dict):
        if np.random.rand() > self.p:
            return data

        if 'events' in data:
            data['events']['x'] = self.width - 1 - data['events']['x']

        if 'image' in data:
            # [H, W, C] を左右反転
            data['image'] = data['image'][:, ::-1, :]

        if 'tracks' in data:
            data['tracks']['x'] = self.width - 1 - (data['tracks']['x'] + data['tracks']['w'])

        return data

class Crop:
    def __init__(self, min_val: List[float], max_val: List[float]):
        self.min_ratio = min_val
        self.max_ratio = max_val

    def init(self, height, width):
        self.min_px = np.array([_scale_and_clip(m, s) for m, s in zip(self.min_ratio, [width, height])])
        self.max_px = np.array([_scale_and_clip(m, s) for m, s in zip(self.max_ratio, [width, height])])

    def __call__(self, data: dict):
        if 'events' in data:
            data['events'] = _crop_events(data['events'], self.min_px, self.max_px)
        if 'image' in data:
            data['image'] = _crop_image(data['image'], self.min_px, self.max_px)
        if 'tracks' in data:
            data['tracks'] = _crop_bbox(data['tracks'], self.min_px, self.max_px)
        return data

class RandomZoom:
    def __init__(self, zoom: List[float], subsample=False):
        self.zoom_range = zoom
        self.subsample = subsample
        self.image_bg = None
        self.height = None
        self.width = None

    def _subsample_events(self, events, zoom_factor, count):
        pos_zoom = np.stack([events['x'], events['y']], axis=-1).astype(np.float64)
        mask = np.zeros(len(events), dtype=bool)
        polarity = events['p'].astype(np.float64)
        
        _subsample(pos_zoom, polarity, mask, count, threshold=1/(float(zoom_factor)**2))

        events = events[mask]
        events['x'] = pos_zoom[mask, 0].astype(events['x'].dtype)
        events['y'] = pos_zoom[mask, 1].astype(events['y'].dtype)
        return events

    def init(self, height, width):
        self.height = height
        self.width = width
        self.image_bg = np.zeros((height, width, 3), dtype=np.uint8)
        self._count = np.zeros((height + 1, width + 1), dtype=np.float32)

    def __call__(self, data: dict):
        zoom = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        new_w = int(np.ceil(self.width * zoom))
        new_h = int(np.ceil(self.height * zoom))

        if 'events' in data:
            data['events']['x'] = ((data['events']['x'] - self.width // 2) * zoom + self.width // 2)
            data['events']['y'] = ((data['events']['y'] - self.height // 2) * zoom + self.height // 2)

            if self.subsample and zoom < 1:
                data['events'] = self._subsample_events(data['events'], zoom, count=self._count.copy())

        if 'image' in data:
            bg_copy = self.image_bg.copy() if zoom < 1 else None
            data['image'] = _resize_image(data['image'], width=new_w, height=new_h, bg=bg_copy)

        if 'tracks' in data:
            data['tracks']['w'] *= zoom
            data['tracks']['h'] *= zoom
            data['tracks']['x'] = ((data['tracks']['x'] - self.width // 2) * zoom + self.width // 2)
            data['tracks']['y'] = ((data['tracks']['y'] - self.height // 2) * zoom + self.height // 2)

        return data

class RandomCrop:
    def __init__(self, size: List[float] = [0.75, 0.75], p=0.5):
        self.size_ratio = size
        self.p = p

    def init(self, height, width):
        self.size_px = np.array([_scale_and_clip(s, ss) for s, ss in zip(self.size_ratio, [width, height])])
        self.left_max = np.array([width, height]) - self.size_px

    def __call__(self, data: dict):
        if np.random.rand() > self.p:
            return data

        left = (np.random.rand(2) * self.left_max).astype(int)
        right = left + self.size_px

        if 'events' in data:
            data['events'] = _crop_events(data['events'], left, right)
        if 'image' in data:
            data['image'] = _crop_image(data['image'], left, right)
        if 'tracks' in data:
            data['tracks'] = _crop_bbox(data['tracks'], left, right)

        return data

class RandomTranslate:
    def __init__(self, size: List[float]):
        self.size_ratio = np.array(size, dtype=float)

    def init(self, height, width):
        self.height = height
        self.width = width
        self.size_px = np.array([_scale_and_clip(s, ss) for s, ss in zip(self.size_ratio, [width, height])])

    def pad(self, image, bg):
        px = (bg.shape[1] - image.shape[1]) // 2
        py = (bg.shape[0] - image.shape[0]) // 2
        bg[py:py + image.shape[0], px:px + image.shape[1]] = image
        return bg

    def __call__(self, data: dict):
        move_px = (self.size_px * (np.random.rand(2) * 2 - 1)).astype(int)

        if 'events' in data:
            data['events']['x'] += move_px[0]
            data['events']['y'] += move_px[1]

        if 'image' in data:
            bg = np.zeros((self.height + 2 * self.size_px[1], self.width + 2 * self.size_px[0], 3), dtype=np.uint8)
            padded_img = self.pad(data['image'], bg)
            y_start = self.size_px[1] - move_px[1]
            x_start = self.size_px[0] - move_px[0]
            data['image'] = padded_img[y_start : y_start + self.height, 
                                       x_start : x_start + self.width]

        if 'tracks' in data:
            data['tracks']['x'] += move_px[0]
            data['tracks']['y'] += move_px[1]

        return data

# ==========================================
# Setup and Augmentations
# ==========================================
def init_transforms(transforms, height, width):
    for t in transforms.transforms:
        if hasattr(t, "init"):
            t.init(height=height, width=width)

class Augmentations:
    def __init__(self, config):
        self.transform_training = Compose([
            RandomHFlip(p=config.aug_p_flip),
            RandomCrop(size=config.random_crop_size, p=config.random_crop_p),
            RandomZoom(zoom=[1.0, config.aug_zoom], subsample=True),
            RandomTranslate([config.aug_trans, config.aug_trans]),
            Crop([0.0, 0.0], [1.0, 1.0]),
        ])
        
        self.transform_testing = Compose([
            Crop([0.0, 0.0], [1.0, 1.0]),
        ])