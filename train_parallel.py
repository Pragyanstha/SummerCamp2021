from __future__ import division
from __future__ import print_function

import numpy as np

import os
from numpy.random import shuffle
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from tensorboardX import SummaryWriter
from tqdm import tqdm


from utils import *
from models import *

from isao import Isao
from config import parse
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train(noise,generator, discriminator, optim_gen, optim_dis,
        schedulers, train_loader, args, rank = 0):
    device = torch.device(f'cuda:{rank}')
    if rank == 0:
        writer=SummaryWriter(f'logs/{args.expname}')
    gen_step = 0
    global_steps = 0
    with tqdm(args.epochs*len(train_loader), position=rank) as pbar:
        for epoch in range(args.epochs):
            for index, minibatch in enumerate(train_loader):

                img = minibatch['img']

                real_imgs = img.to(device)

                noise = torch.from_numpy(np.random.normal(0, 1, (img.shape[0], args.latent_dim))).float().to(device)

                optim_dis.zero_grad()
                real_valid=discriminator(real_imgs)
                fake_imgs = generator(noise).detach()

                fake_valid = discriminator(fake_imgs)

                if args.loss == 'hinge':
                    loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
                elif args.loss == 'wgangp_eps':
                    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), args.phi)
                    loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 / (args.phi ** 2)         

                loss_dis.backward()
                optim_dis.step()
                if rank == 0:
                    writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

                if global_steps % args.n_critic == 0:

                    optim_gen.zero_grad()
                    if schedulers:
                        gen_scheduler, dis_scheduler = schedulers
                        g_lr = gen_scheduler.step(global_steps)
                        d_lr = dis_scheduler.step(global_steps)
                        writer.add_scalar('LR/g_lr', g_lr, global_steps)
                        writer.add_scalar('LR/d_lr', d_lr, global_steps)

                    gener_noise = torch.from_numpy(np.random.normal(0, 1, (args.gener_batch_size, args.latent_dim))).float().to(device)

                    generated_imgs= generator(gener_noise)
                    fake_valid = discriminator(generated_imgs)

                    gener_loss = -torch.mean(fake_valid).to(device)
                    gener_loss.backward()
                    optim_gen.step()
                    if rank == 0:
                        writer.add_scalar("gener_loss", gener_loss.item(), global_steps)

                    gen_step += 1
                
                pbar.set_description(f'Epoch: {epoch}, batch: {global_steps%len(train_loader)}/{len(train_loader)}, D Loss: {loss_dis.item():07.4f}, G Loss: {gener_loss.item():07.4f}')
                pbar.update()
                global_steps += 1

                if gen_step and index % 100 == 0 and rank == 0:
                    sample_imgs = generated_imgs[:32].detach().cpu()
                    save_image(sample_imgs, f'generated_images/{args.expname}/img_{epoch}_{index % len(train_loader)}.jpg', nrow=5, normalize=True, scale_each=True)            
                    sample_imgs = (sample_imgs - sample_imgs.min())/(sample_imgs.max() - sample_imgs.min())
                    writer.add_images('Generated Samples', sample_imgs, global_steps)
                    # tqdm.write("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                    #     (epoch+1, index % len(train_loader), len(train_loader), loss_dis.item(), gener_loss.item()))
                    checkpoint = {'epoch':epoch }
                    checkpoint['generator_state_dict'] = generator.module.state_dict()
                    checkpoint['discriminator_state_dict'] = discriminator.module.state_dict()
                    save_checkpoint(checkpoint, output_dir=f'checkpoint/{args.expname}', filename='latest')

            if rank == 0:
                checkpoint = {'epoch':epoch }
                checkpoint['generator_state_dict'] = generator.state_dict()
                checkpoint['discriminator_state_dict'] = discriminator.state_dict()
                save_checkpoint(checkpoint, output_dir='checkpoint', filename=f'epoch_{epoch}')


def main(rank, world_size):
    setup(rank, world_size)
    generator = Generator(initial_size = args.initial_size, latent_dim = args.latent_dim, dim = args.dim).to(f'cuda:{rank}')
    discriminator = Discriminator(diff_aug=args.diff_aug, patch_size=args.patch_size, dim=args.dim).to(f'cuda:{rank}')


    if args.checkpoint != None and os.path.isfile(f'checkpoint/{args.expname}/{args.checkpoint}'):
        print(f'Loaded {args.checkpoint}')
        map_location = {'cuda:0':f'cuda:{rank}'}
        checkpoint = torch.load(f'checkpoint/{args.expname}/{args.checkpoint}')
        generator.load_state_dict(checkpoint['generator_state_dict'], map_location)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], map_location)

    generator = DDP(generator, device_ids=[rank])#,device = device))
    discriminator = DDP(discriminator, device_ids=[rank])
    generator.apply(inits_weight)
    discriminator.apply(inits_weight)

    # Initialize datasets
    train_set = Isao(args.data_dir, use_label=False, resize=(64,64))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, 
        sampler = torch.utils.data.DistributedSampler(train_set, shuffle=True, rank = rank))
    
    # Initialize Optimizer
    if args.optim == 'Adam':
        optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(args.beta1, args.beta2))

        optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),lr=args.lr_dis, betas=(args.beta1, args.beta2))
        
    elif args.optim == 'SGD':
        optim_gen = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()),
                    lr=args.lr_gen, momentum=0.9)

        optim_dis = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()),
                    lr=args.lr_dis, momentum=0.9)

    elif args.optim == 'RMSprop':
        optim_gen = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

        optim_dis = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

    gen_scheduler = LinearLrDecay(optim_gen, args.lr_gen, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(optim_dis, args.lr_dis, 0.0, 0, args.max_iter * args.n_critic)


    print("optim:",args.optim)
    lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None

    
    generator = generator.train()
    discriminator = discriminator.train()
    
    # Training loop
    train(noise, generator, discriminator, optim_gen, optim_dis, lr_schedulers, 
    train_loader, args, rank)

    print("Hurray! finished all epochs.")
    cleanup()


args = parse()
if __name__ == '__main__':
    os.makedirs(f'checkpoint/{args.expname}', exist_ok=True)
    os.makedirs(f'generated_images/{args.expname}', exist_ok=True)
    world_size = 2
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)