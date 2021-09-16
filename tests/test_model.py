import torch
import numpy as np
import sys
sys.path.append('./')
import models


if __name__ == '__main__':
    netG = models.Generator(initial_size = 8, latent_dim = 1024, dim = 512).cuda(1)#,device = device))
    netD = models.Discriminator(diff_aug='translation,cutout,color', patch_size=2, dim=256).cuda(1)

    # Test Generator
    noise = torch.from_numpy(np.random.normal(0, 1, (1, 1024))).float().cuda(1)
    img = netG(noise)
    valid = netD(img)
    print(img.shape)
    print(valid)