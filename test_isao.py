import torch
from torch.utils.data import DataLoader
import isao

def main():
    train_dataset = isao.Isao('./data/preprocessed', use_label=True, resize=(64,64))

    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    for batch in train_dataloader:
        print(batch['img'].shape)



if __name__ == '__main__':
    main()