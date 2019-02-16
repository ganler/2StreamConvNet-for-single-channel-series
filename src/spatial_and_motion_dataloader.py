import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np

class sm_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, mode, transform=None):
        tmp_df = pd.read_csv(csv_path) # 将csv的位置作形参
        self.transform = transform
        if mode == 'spatial' or mode == 'motion':
            self.channel_0 = tmp_df[mode[0]+'_frame_0']
            self.channel_1 = tmp_df[mode[0] + '_frame_1']
            self.channel_2 = tmp_df[mode[0] + '_frame_2']
        else:
            raise ValueError('mode = motion or spatial')
        self.y_train = tmp_df['tags'].astype(np.float32)

    def __getitem__(self, index):
        img_0 = Image.open(self.channel_0[index]).convert('L') # 对于灰度图是'L'
        img_1 = Image.open(self.channel_1[index]).convert('L')
        img_2 = Image.open(self.channel_2[index]).convert('L')

        # print(img_0)
        if self.transform is not None:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        img = torch.cat([img_0, img_1, img_2], 0)

        label = torch.from_numpy(
            np.array(self.y_train[index])
        		).long()
        return img, label

    def __len__(self):
        return len(self.y_train.index)