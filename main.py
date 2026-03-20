import warnings
from pathlib import Path

import joblib
import segmentation_models_pytorch as smp
import torch
from sklearn.ensemble import GradientBoostingRegressor
from torchvision import transforms

from src.hbef import HBEF, GBRSignedEP, Resnet50UnetSegLayer2d

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    try:
        unet = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
        unet.load_state_dict(torch.load('artifacts/resnet50-unet:v1/resnet50.pth', map_location=torch.device('cpu')))
        gbr: GradientBoostingRegressor = joblib.load('artifacts/error-predictor:v4/gbr.joblib')

        hbef = HBEF(
            seglayer=Resnet50UnetSegLayer2d(
                cnn=unet, 
                transf=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ]),
                volume_estimation_method='bullet',
                pixel_spacing=0.1
            ),
            signedep=GBRSignedEP(
                model=gbr,
                epsilon=1e-5
            )
        )

        pred = hbef.predict(
            test=list(Path('test/').glob("*.avi")),
            output_dir = Path('inference'),
            verbose=True,
            overlay_color=(114, 6, 20)
        )

        print(pred)

    except FileNotFoundError:
        print('no artifacts found :( use download_artifacts.py before running this script')