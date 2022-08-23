import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()

        for i in tqdm(range(args.classes)):
            id = torch.randn(1, int(args.latent / 2), device=device)
            id = id.repeat(args.images, 1)
            other = torch.randn(args.images, int(args.latent / 2), device=device)
            z = torch.cat((id, other), 1)
            sample = g_ema(
                z, one_id=True, truncation=args.truncation, truncation_latent=mean_latent
            )

            for j in range(args.images):
                utils.save_image(
                    sample[j],
                    f"images/{str(i).zfill(6)}_{str(j).zfill(3)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=1000,
        help="number of classes to be generated",
    )
    parser.add_argument(
        "--images", type=int, default=128, help="number of images to be generated for each class"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "ckpt",
        type=str,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier of the generator",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
