import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import Datasets
import Models

class Inference:
    def __init__(self, model, input_size, model_path, ensemble = False, nick_name='th'):
        self.model = model
        self.input_size = input_size
        self.model_path = model_path
        self.ensemble = ensemble
        self.nickname = nick_name

        test_dir = '/opt/ml/input/data/eval'
        image_dir = os.path.join(test_dir, 'images')
        self.dataset = Datasets.TestDataset
        self.transform = Datasets.transform_test(input_size=self.input_size)
        self.submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
        self.image_paths = [os.path.join(image_dir, img_id) for img_id in self.submission.ImageID]


    def infer(self):
        print(f"Inference {self.model_path}")
        print("*"*20)
        # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
        dataset = self.dataset(self.image_paths, self.transform)

        loader = DataLoader(
            dataset,
            shuffle=False,
            num_workers=4,
        )

        device = torch.device('cuda')
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(device)
        self.model.eval()

        results = []
        for images in tqdm(loader):
            with torch.no_grad():
                images = images.to(device)
                pred = self.model(images)
                if self.ensemble:
                    results.extend(pred.cpu().numpy())
                else:
                    results.extend(pred.argmax(dim=-1).cpu().numpy())
        if not(self.ensemble):
            base = '../input/data/eval/'
            self.submission['ans'] = results
            save_name = self.model_path.split('/')[-2:]
            save_name = save_name[0]+save_name[1].split('.')[0]
            self.submission.to_csv(base+self.nickname+save_name+'.csv')
        return np.asarray(results)


def albumentation(size=(224, 224), use_filp=True, use_rotate=True, use_blur=True,
                  use_noise=True, use_normalize=True, use_CLAHE=True,
                  use_invert=True, use_equalize=True, use_posterize=True,
                  use_soloarize=True, ues_jitter=True, use_Brightness=True,
                  use_Gamma=True, use_brightcontrast=True, use_cutout=True,
                  use_totensor=True):
    random_crop = A.RandomCrop(height=size[0], width=size[1], p=1)
    random_resizecrop = A.RandomResizedCrop(height=size[0], width=size[1])
    random_filp = A.HorizontalFlip(p=0.5)
    random_blur = A.GaussianBlur()
    random_noise = A.GaussNoise()
    random_rotate = A.Rotate(limit=45, p=0.65)
    random_CLAHE = A.CLAHE(p=0.4)
    random_invert = A.InvertImg(always_apply=False)
    random_equalize = A.Equalize(always_apply=False)
    random_posterize = A.Posterize(always_apply=False)
    random_solarize = A.Solarize(always_apply=False)
    random_jitter = A.ColorJitter(always_apply=False)
    random_Brightness = A.RandomBrightness(always_apply=False)
    random_Gamma = A.RandomGamma(always_apply=False)
    random_brightcontrast = A.RandomBrightnessContrast(always_apply=False)
    random_cutout = A.Cutout(max_h_size=int(size[0]*0.1), max_w_size=int(size[1]*0.1),always_apply=False)

    normalize = A.Normalize(always_apply=True)

    transforms = np.array([random_crop, random_resizecrop, random_filp,
                           random_invert, random_equalize, random_posterize,
                           random_solarize, random_jitter, random_Brightness,
                           random_Gamma, random_brightcontrast, random_cutout,
                           random_rotate, random_CLAHE, random_blur,
                           random_noise,normalize, ToTensorV2()])

    transform_mask = [False, True, use_filp,
                      use_invert, use_equalize, use_posterize,
                      use_soloarize, ues_jitter, use_Brightness,
                      use_Gamma, use_brightcontrast, use_cutout,
                      use_rotate, use_CLAHE, use_blur,
                      use_noise, use_normalize, use_totensor ]
    transforms = A.Compose(transforms[transform_mask])
    return transforms
