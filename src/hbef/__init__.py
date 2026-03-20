from __future__ import annotations

__version__ = "0.0.1"

import typing as tp
from pathlib import Path

import cv2
import numpy as np

from src.hbef.seglayer2d import Resnet50UnetSegLayer2d, Seg2dInfer, SegLayer2d
from src.hbef.signedep import GBRSignedEP, SignedEP

__all__ = ['Resnet50UnetSegLayer2d', 'GBRSignedEP']


class HBEF:
    def __init__(self, seglayer: SegLayer2d, signedep: SignedEP) -> None:
       self._seglayer = seglayer
       self._signedep = signedep

    def predict(
        self, 
        file_path: tp.Union[str, Path], 
        output_dir: tp.Union[str, Path] = Path('inference'),
        verbose: bool = False, 
        overlay_color: tp.Tuple[int, int, int] = (114, 6, 20)   
    ) -> tp.Tuple[np.float16, np.float16]:
        file_path = Path(file_path)
        cap: cv2.VideoCapture = self._load_video(file_path)
        frames, fps, w, h = self._get_frame_sequence(cap)
        seginfer: Seg2dInfer = self._seglayer.predict(frames)
        signed_error = self._signedep.predict(seginfer)

        if verbose:
            output_dir.mkdir(exist_ok=True)
            video_writer = cv2.VideoWriter(str(output_dir / file_path.name), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
            for idx in range(0, len(frames)):
                fvis = cv2.cvtColor(frames[idx], cv2.COLOR_GRAY2BGR)
                mask_overlay = np.zeros_like(fvis)
                mask_overlay[seginfer.segmasks[idx] == 1] = overlay_color
                blended = cv2.addWeighted(fvis, 1.0, mask_overlay, 1.0, 0)
                video_writer.write(blended)
            video_writer.release()
            cap.release()

        return seginfer.ef, signed_error

    def _load_video(self, file_path: Path) -> cv2.VideoCapture:
        if not file_path.is_file():
            raise FileNotFoundError
        return cv2.VideoCapture(file_path)

    def _get_frame_sequence(self, cap: cv2.VideoCapture) -> tp.Tuple[tp.List[np.array], float, int, int] :
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            return frames, fps, w, h
        finally:
            cap.release()