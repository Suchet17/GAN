from datetime import datetime
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image as savefig
from torchvision import transforms
from model import DiscriminativeNetwork, GenerativeNetwork, init_weights
import torchvision.datasets as datasets
from config import (z_dim, img_channels, batch_size,
                    learning_rate_d, learning_rate_g, num_epochs,
                    features_disc, features_gen)

if __name__ == '__main__':
    data = "CIFAR10"
    version = 10
    try:
        os.mkdir(f"GAN/{data}/logs/try{version}")
    except FileExistsError:
        if os.path.exists(f"GAN/{data}/logs/try{version}/output.log"):
            print("Change Version Number")
            raise SystemExit
    num_images = 10000
    num_batches = int(num_images/batch_size)
    k = num_batches // 5
    now = datetime.now
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        d1evice = torch.device('cpu')

    # Setup models
    disc = DiscriminativeNetwork(img_channels, features_disc).to(device)
    gen  = GenerativeNetwork(z_dim, img_channels, features_gen).to(device)
    init_weights(disc)
    init_weights(gen)

    #Setup training data
    dataset = datasets.CIFAR10(root = "GAN/CIFAR10/data",
                                    train=False, download = False,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop((28,28)),
                                        transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle = True, num_workers = 4)

    # Setup trainers
    optim_d = Adam(disc.parameters(), lr=learning_rate_d, betas =(0.5, 0.999))
    optim_g = Adam(gen.parameters(),  lr=learning_rate_g, betas =(0.5, 0.999))
    criterion = nn.BCELoss()

    # Evaluation on same input latent vector
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device) # Fixed Random Seed

    gen.train()
    disc.train()
    f = open(f"GAN/{data}/logs/try{version}/output.log", 'w')
    print("No BatchNorm for Discriminator", file=f)
    print(f"Learning Rate Gen = {learning_rate_g}\tLearning Rate Disc = {learning_rate_d}", file=f, flush=True)
    print("Start Training")
    #Training Loop
    for epoch in range(num_epochs):
        for batch_index , (real, _ ) in enumerate(dataloader):
            real = real.to(device)
            # print("batch_size =", real.shape[0])

            z = torch.randn((batch_size, z_dim, 1, 1)).to(device)
            fake = gen(z)
            # print(f"{epoch}\t{batch_index}\tCP 1", end = '\t')
            # print(torch.cuda.memory_summary(), file=f, flush=True)

            # Train Discriminator: maximize log(D(real)) + log(1-D(G(z)))

            disc_real = disc(real).reshape(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake)/2

            disc.zero_grad()
            lossD.backward()
            optim_d.step()
            # print("CP 2", end = '\t')
            # print(torch.cuda.memory_summary(), file=f, flush=True)

            # Train Generator: maximize log(D(G(z)))
            output = disc(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            optim_g.step()
            # print("CP 3")
            # print(torch.cuda.memory_summary(), file=f, flush=True)

            if batch_index % k == 0:
                print(f"{datetime.now().time()}\t"+
                      f"Epoch [{epoch+1}/{num_epochs}]\t"+
                      f"Batch [{batch_index+1}/{len(dataloader)}]\t"+
                      f"Loss D: {lossD:.4f}\tloss G: {lossG:.4f}",
                      file = f, flush=True)

                fake = gen(fixed_noise).to(torch.device('cpu'))
                savefig(fake, f"GAN/{data}/logs/try{version}/Image{epoch+1}_{batch_index//k+1}.png")
                print(f"Saved Image{epoch+1}_{batch_index//k+1}.png")
            # print(torch.cuda.memory_summary(), file=f, flush=True)
    f.close()
