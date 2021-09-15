from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import numpy as np
import os
from torchvision import transforms as transforms

class Isao(Dataset):
    def __init__(self, data_dir, use_label, resize = None):
        self.data_dir = data_dir
        self.use_label = use_label
        self.files = glob.glob(data_dir + '/**/*.jpg', recursive=True)
        self.files = [f.replace('\\n', '/n') for f in self.files]
        self.labels = self.get_label(self.files)
        self.label_map = np.eye(len(self.labels))
        if resize != None:
            self.transform = transforms.Compose(
                [transforms.Resize(resize),
                transforms.ToTensor()]
            ) 
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor()]
            )
    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        filepath = self.files[idx]
        img = Image.open(filepath)
        label = self.get_label(filepath)
        label_idx = self.labels.index(label)
        label_one_hot = self.label_map[label_idx]

        sample = {'img': self.transform(img), 'label_name': [label], 'label_one_hot': label_one_hot}
        return sample


    def __str__(self):
        desc = []
        for i in range(len(self.labels)):
            s = f'{self.labels[i]} : {str(self.label_map[i])}'
            desc.append(s)
        
        return '\n'.join(desc)

    def get_label(self, files):
        if type(files) == list:
            folders = os.listdir(self.data_dir)
            label = []
            for folder in folders:
                label.append('-'.join(folder.split('-')[1:]))
            return label
        else:
            folder_name = files.split('/')[2]
            label = '-'.join(folder_name.split('-')[1:])
            return label