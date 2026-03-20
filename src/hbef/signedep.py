import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.hbef.seglayer2d import Seg2dInfer

T = tp.TypeVar('T')
class SignedEP(ABC, tp.Generic[T]):
    def __init__(self, model: T) -> None:
        self._model = model

    @abstractmethod
    def predict(seginfer: Seg2dInfer) -> np.float16:
        ...

@dataclass
class GBRInput:
    lv_volume_ratio: np.float16
    lv_length_ratio: np.float16
    seg_dice_score_std: np.float16

    def to_numpy(self) -> np.ndarray:
        return np.array([self.lv_volume_ratio, self.lv_length_ratio, self.seg_dice_score_std]).reshape(1,-1)


class GBRSignedEP(SignedEP[GradientBoostingRegressor]):
    def __init__(self, model: GradientBoostingRegressor, epsilon: float = 1e-5) -> None:
        super().__init__(model)
        self._epsilon = epsilon

    def predict(self, seginfer: Seg2dInfer) -> np.float16:
        input = self._build_input(seginfer)
        return np.float16(self._model.predict(input.to_numpy()))

    def _build_input(self, seginfer: Seg2dInfer) -> GBRInput:
        lv_volume_ratio = self._get_lv_volume_ratio(seginfer.lv_volumes)
        lv_length_ratio = self._get_lv_length_ratio(seginfer.lv_lengths)
        seg_dice_score_std = self._get_seg_dice_score_std(seginfer.segmasks)
        return GBRInput(lv_volume_ratio, lv_length_ratio, seg_dice_score_std)

    def _get_lv_volume_ratio(self, lv_volumes: tp.List[np.float16]) -> np.float16:
        edv, esv = np.max(lv_volumes), np.min(lv_volumes)
        return np.float16(esv / edv) if edv > 0 else np.float16(0.0)

    def _get_lv_length_ratio(self, lv_lengths: tp.List[np.float16]) -> np.float16:
        edl, esl = np.max(lv_lengths), np.min(lv_lengths)
        return np.float16(esl / edl) if edl > 0 else np.float16(0.0)

    def _get_seg_dice_score_std(self, segmasks: tp.List[np.ndarray]) -> np.float16:
        dice_scores: tp.List[np.float16] = []
        prev_segm = None
        for curr_segm in segmasks:
            if prev_segm is not None:
                denom = prev_segm.sum() + curr_segm.sum()
                inter = np.logical_and(prev_segm, curr_segm).sum()
                dice = (2.0 * inter) / (denom + self._epsilon)
                dice_scores.append(dice)
            prev_segm = curr_segm
        return np.std(dice_scores)

    