import argparse
import math
import random
import os
import pretrainedmodels
import pretrainedmodels.utils
import torchvision.models as models
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import pickle
try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
from dataset import IrisDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class iris_classifier(nn.Module):
    def __init__(self):
        super(iris_classifier, self).__init__()

        self.resNet50 = models.resnet50(pretrained=True)
        modules = list(self.resNet50.children())[:-1]
        self.resNet50 = torch.nn.Sequential(*modules)
        self.resNet50.flat = torch.nn.Flatten()

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, img1, img2):

        out1 = self.resNet50(img1)
        out2 = self.resNet50(img2)

        out1 = F.relu(self.fc1(out1))
        out2 = F.relu(self.fc1(out2))

        out = torch.cat((out1, out2), 1)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach()


def make_noises(batch, latent_dim, device, one_id):
    if one_id:
        id = torch.randn(1, int(latent_dim / 2), device=device)
        id = id.repeat(batch, 1)
        other = torch.randn(batch, int(latent_dim / 2), device=device)
        return torch.cat((id, other), 1)
    else:
        return torch.randn(batch, latent_dim, device=device)


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, irisnet, device):

    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0.0

    id_criterion = torch.nn.L1Loss().to(device)
    r1_loss = torch.tensor(0.0, device=device)
    id_loss_f = torch.tensor(0.0, device=device)
    id_loss_t = torch.tensor(0.0, device=device)
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (16 / (5 * 1000))
    pairs = list(range(args.batch))[1:]
    pairs.append(0)

    for idx in pbar:

        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noises = make_noises(args.batch, args.latent, device, one_id=False)
        fake_img = generator(noises, one_id=False)

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noises = make_noises(args.batch, args.latent, device, one_id=False)
        fake_img = generator(noises, one_id=False)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        if i >= args.id_delay:

            noises_t = make_noises(args.batch, args.latent, device, one_id=True)
            fake_img_t = generator(noises_t, one_id=True)

            id_features_f = irisnet(fake_img, fake_img[pairs])
            pred_f = torch.sigmoid(id_features_f)
            id_loss_f = id_criterion(pred_f, torch.zeros_like(pred_f))


            id_features_t = irisnet(fake_img_t, fake_img_t[pairs])
            pred_t = torch.sigmoid(id_features_t)
            id_loss_t = id_criterion(pred_t, torch.ones_like(pred_t))


        loss_dict["id_f"] = id_loss_f
        loss_dict["id_t"] = id_loss_t

        g_id_loss = g_loss + id_loss_f + id_loss_t

        generator.zero_grad()
        g_id_loss.backward()
        g_optim.step()

        g_regularize = (i % args.g_reg_every == 0 and i < args.id_delay)

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noises(path_batch_size, args.latent, device, one_id=False)
            fake_img, latents = generator(noise, one_id=False, return_latents=True)

            path_loss, mean_path_length = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()


        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        id_loss_f_val = loss_reduced["id_f"].mean().item()
        id_loss_t_val = loss_reduced["id_t"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"id_f: {id_loss_f_val:.4f}; id_t: {id_loss_t_val:.4f}; "
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "generator": g_loss_val,
                        "discriminator": d_loss_val,
                        "r1": r1_val,
                        "id_false": id_loss_f_val,
                        "id_true": id_loss_t_val,
                    }
                )

            if i % 10000 == 0:

                with torch.no_grad():
                    g_ema.eval()
                    sample_f = make_noises(args.n_sample, args.latent, device, one_id=False)
                    sample = g_ema(sample_f, one_id=False)
                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(7)}f.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    sample_t = make_noises(args.n_sample, args.latent, device, one_id=True)
                    sample = g_ema(sample_t, one_id=True)
                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(7)}t.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"checkpoint/{str(i).zfill(7)}.pt",
                )



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Iris GANs trainer")

    parser.add_argument(
        "path", type=str, help="path to the iris dataset"
    )
    parser.add_argument(
        "--irisnet", type=str,
        default='./verification_net.pth',
        help="path to the pretrained iris verification net"
    )
    parser.add_argument(
        "--iter", type=int, default=1500000, help="total training iterations"
    )
    parser.add_argument(
        "--id_delay", type=int, default=500000, help="number of iterations delayed in identity training"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch size for each gpu"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=1, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=8,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0025, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier factor for the model.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--distributed", action="store_true", help="apply distributed training"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    args = parser.parse_args()

    if args.distributed:
        n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    irisnet = iris_classifier().to(device)
    irisnet.eval()

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    print("loading pretrained iris verification net '{}'".format(args.irisnet))
    ckpt_iris = torch.load(args.irisnet, map_location=lambda storage, loc: storage)
    irisnet.load_state_dict(ckpt_iris["state_dict"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = IrisDataset(args.path, transform)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="Iris GANs")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, irisnet, device)
