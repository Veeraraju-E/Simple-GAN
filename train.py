import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Discriminator, Generator

# GANs are quite sensitive to hyperparameters, like really sensitive :\
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 5e-4
LATENT_DIM = 64
IMAGE_DIM = 784
BATCH_SIZE = 32
EPOCHS = 100

discriminator = Discriminator(IMAGE_DIM)
generator = Generator(latent_dim=LATENT_DIM, output_dim=IMAGE_DIM)
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

train_dataset = datasets.MNIST(root='datasets/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST(root='datasets/', train=False, transform=transformation, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optim_disc = optim.Adam(discriminator.parameters(), lr=LR)
optim_gen = optim.Adam(generator.parameters(), lr=LR)

loss_fn = nn.BCELoss()


def train(loader):
    for epoch in range(EPOCHS):
        for batch_idx, (x, _) in tqdm(enumerate(loader)):
            x = x.view(-1, 784).to(DEVICE)

            # now we have to train the discriminator and the generator separately
            # generator generates images from Gaussian noise
            # discriminator samples from the real set of images and compares the generated fake image and the real image
            # the comparison would be between the encodings of the generator and the discriminator

            # x has samples belonging to a particular batch
            # print(x.shape) --> [32, 784]
            real_img = discriminator(x).view(-1)  # real_img is the D(x)
            # print(real_img.shape) --> [32]
            noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            fake_img_gen = generator(noise)  # fake_img_gen.shape --> [32, 784]
            # print(fake_img_gen.shape)
            fake_img = discriminator(fake_img_gen).view(-1)  # fake_img_gen is the G(z), fake_img is the D(G(z))

            # loss_disc_real = torch.log(real_img) --> for one sample, for all samples, we need
            # expectation overall x ~ p_data(x) --> we can use BCELoss here, between D(x) and

            # loss_disc_fake = 1 - torch.log(fake_img) --> for one sample, for all samples, we need
            # expectation over all z ~ p_z(z)

            # total loss for discriminator -> log(D(x)) + log(1 - D(G(z))) to be **maximized** wrt params of disc
            # first part would be the same as minimizing the BCELoss between torch.ones_like(D(x)) and D(x) itself, so that
            # the 2nd term in the BCELoss would become 0, and we get the log(D(x))
            # Similar explanation for the 2nd part of loss
            loss_disc_total = (loss_fn(real_img, torch.ones_like(real_img)) + loss_fn(fake_img, torch.zeros_like(fake_img)))/2

            optim_disc.zero_grad()
            loss_disc_total.backward(retain_graph=True)  # a bit tricky, but just to avoid losing all gradients computed
            # for the G(z) in this step, as it would be used in the loss fn of the generator too
            optim_disc.step()

            # total loss for the generator -> log(1 - D(G(z))) only, and to be **minimized** wrt params of generator
            # loss_gen_total = -loss_fn(fake_img, torch.zeros_like(fake_img)) --> leads to saturating gradients
            # therefore better to maximize log(D(G(z))
            output = discriminator(fake_img_gen).view(-1)
            loss_gen_total = loss_fn(output, torch.ones_like(fake_img))

            optim_gen.zero_grad()
            loss_gen_total.backward()
            optim_gen.step()

            if batch_idx == 31:
                print(f'{epoch}/{EPOCHS} loss disc = {loss_disc_total:.4f} loss gen = {loss_gen_total:.4f}')

                with torch.no_grad():
                    final_generated_img = generator(noise).reshape(-1, 1, 28, 28)
                    save_image(final_generated_img, f'generated_{epoch}_{batch_idx}.png')


if __name__ == '__main__':
    train(train_loader)
