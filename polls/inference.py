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
