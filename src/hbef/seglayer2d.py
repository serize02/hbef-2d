import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy.spatial.distance import pdist
from torchvision import transforms


@dataclass
class Seg2dInfer:
    segmasks: tp.List[np.ndarray]
    lv_areas: tp.List[np.uint16]
    lv_lengths: tp.List[np.float16]
    lv_volumes: tp.List[np.float16]
    ef: np.float16

T = tp.TypeVar('T')
TransformType = tp.Callable[[np.ndarray], torch.Tensor] # alias
class SegLayer2d(ABC, tp.Generic[T]):
    def __init__(
        self, 
        cnn: T, 
        transf: TransformType = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        volume_estimation_method: str = 'bullet',
        pixel_spacing: float = 0.1

    ) -> None:
        self._cnn = cnn
        self._transforms = transf
        self._volume_estimation_method = volume_estimation_method
        self._pixel_spacing = pixel_spacing

    def predict(self, frames: tp.List[np.ndarray]) -> Seg2dInfer:
        segmasks: tp.List[np.ndarray] = self._get_segmentation_masks(frames)
        lv_areas: tp.List[np.uint16] = self._get_lv_area(segmasks)
        lv_lengths: tp.List[np.float16] = self._get_lv_length(segmasks)
        lv_volumes: tp.List[np.float16] = self._get_lv_volume(lv_areas, lv_lengths)
        ef: np.float16 = self._get_ef(lv_volumes)
        return Seg2dInfer(segmasks, lv_areas, lv_lengths, lv_volumes, ef)

    @abstractmethod
    def _get_segmentation_masks(self, frames: tp.List[np.ndarray]) -> tp.List[np.ndarray]:
        ...

    @property
    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() 
                else 'mps' if torch.backends.mps.is_available()  # macOS
                else 'cpu')

    def _get_lv_area(self, segmasks: tp.List[np.ndarray]) -> tp.List[np.uint16]:
        return [np.sum(segm).astype(np.uint16) for segm in segmasks]

    def _get_lv_length(self, segmasks: tp.List[np.ndarray]) -> tp.List[np.uint16]:
        return [self._estimate_lv_length(segm) for segm in segmasks]

    def _estimate_lv_length(self, segm: np.ndarray) -> np.float16:
        if segm.size == 0 or np.sum(segm) == 0:
            return np.float16(0.0)
        coords = np.column_stack(np.where(segm == 1))
        if coords.shape[0] < 2:
            return np.float16(0.0)
        distances = pdist(coords, metric='euclidean')
        if distances.size == 0 or not np.all(np.isfinite(distances)):
            return np.float16(0.0)
        return np.float16(np.max(distances))

    def _get_lv_volume(self, lv_areas: tp.List[np.uint16], lv_lengths: tp.List[np.float16]) -> tp.List[np.float16]:
        if self._volume_estimation_method != 'bullet':
            raise RuntimeError('volume_estimate_method not supported')
        return [np.float16(5/6 *  lv_a * lv_l * (self._pixel_spacing ** 3)) for lv_a, lv_l in zip(lv_areas, lv_lengths)]

    def _get_ef(self, lv_volumes: tp.List[np.float16]) -> np.float16:
        edv, esv = np.max(lv_volumes), np.min(lv_volumes)
        return (((edv - esv) / edv) * 100).astype(np.float16)


class Resnet50UnetSegLayer2d(SegLayer2d[smp.decoders.unet.model.Unet]):
    def _get_segmentation_masks(self, frames: tp.List[np.ndarray]) -> tp.List[np.ndarray]:
        self._cnn.to(self._device)
        self._cnn.eval()
        segmasks: tp.List[np.ndarray] = []
        with torch.no_grad():
            for grayf in frames:
                tensor = self._transforms(grayf).unsqueeze(0).to(self._device)
                p = self._cnn(tensor)
                segm = (p > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)
                segmasks.append(segm)
        return segmasks