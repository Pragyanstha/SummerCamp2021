import models
import config
import torch
import numpy as np
import os
from PIL import Image
import torchvision
from tqdm import tqdm
import zipfile

def main(model, opt):
    batch = 20
    os.makedirs('validation', exist_ok=True)
    zipwriter = zipfile.ZipFile('validation/team_c_dayo.zip', 'w')
    total_imgs = 10000
    step = 0
    with tqdm(total_imgs) as pbar:
        for b in range(total_imgs//batch):
            noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (batch, opt.latent_dim)))
            fake_imgs = model(noise)
            for i in range(batch):
                fake_img = torchvision.transforms.functional.to_pil_image(fake_imgs[i])
                fake_imgs = (fake_imgs - fake_imgs.min())/(fake_imgs.max() -fake_imgs.min())
                fake_img.save(f"validation/img_{step}.png")
                zipwriter.write(f"validation/img_{step}.png")
                step += 1
                pbar.set_description(f"Saving : img_{step}.png")
                pbar.update()
            
    zipwriter.close()

if __name__  == '__main__':
    opt = config.parse()
    generator= models.Generator(depth1=5, depth2=4, depth3=2, 
                    initial_size=opt.initial_size, latent_dim=opt.latent_dim, dim=opt.dim, heads=4, mlp_ratio=4, drop_rate=0.5)#,device = device)
    
    checkpoint = torch.load(f'{opt.v}')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f'Loaded {opt.v}')
    generator.to('cuda:0')
    main(generator, opt)