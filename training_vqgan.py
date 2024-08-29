import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
# from lpips import LPIPS
from lpips_tame import LPIPS
# from vqgan import VQGAN
from utils import load_data
from vqgan_tame import VQModel as VQGAN_tame

import wandb

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN_tame(args).to(device=args.device)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)

        # print params count for debugging
        params_count = sum(p.numel() for p in self.vqgan.parameters())
        print(f"Number of parameters: {params_count/1e6:.2f}M")

        self.opt_vq, _ = self.configure_optimizers(args)

        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.quantize.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )

        return opt_vq, None

    @staticmethod
    def prepare_training():
        os.makedirs("results_vqvae", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()

                    vq_loss = perceptual_rec_loss + q_loss

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_vq.step()

                    if i % 50 == 0:
                        with torch.no_grad():
                            # real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results_vqvae", f"{epoch}_{i}.jpg"), nrow=4)

                            wandb.log({
                                "real_fake_images": [wandb.Image(real_fake_images, caption="Real vs Fake images")]
                            })

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        rec_loss=np.round(perceptual_rec_loss.cpu().detach().numpy().item(), 5),
                        q_loss=np.round(q_loss.cpu().detach().numpy().item(), 5),
                    )

                    wandb.log({
                        "epoch": epoch,
                        "VQ_Loss": vq_loss.cpu().detach().numpy().item(),
                        "rec_loss": perceptual_rec_loss.cpu().detach().numpy().item(),
                        "q_loss": q_loss.cpu().detach().numpy().item(),
                    })
                    pbar.update(0)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    # args.dataset_path = r"C:\Users\dome\datasets\flowers"
    args.dataset_path = "jpg"

    wandb.init(project="vqgan_dome", config=args)

    train_vqgan = TrainVQGAN(args)



