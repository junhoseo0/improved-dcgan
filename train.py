import argparse
import torch
import torchvision
import torchvision.transforms as T
from dcgan import Generator, Discriminator

parser = argparse.ArgumentParser("Hyperparameters and Dataset Arguments")
parser.add_argument("--dataset", "-ds", type=str, default="mnist")
parser.add_argument("--batch-size", "-bs", type=int, default=64)
parser.add_argument("--num-noises", "-n", type=int, default=100)
parser.add_argument("--depths", "-d", type=int, default=128)
parser.add_argument("--learning-rate", "-lr", type=float, default=0.0002)
parser.add_argument("--beta-1", "-b1", type=float, default=0.5)
parser.add_argument("--beta-2", "-b2", type=float, default=0.99)
parser.add_argument("--epochs", "-e", type=int, default=10)
args = parser.parse_args()

if args.dataset == "mnist":
    IMAGE_SIZE = 64
    NUM_COLORS = 1

    data = torchvision.datasets.MNIST(
        "../Datasets/MNIST_PyTorch",
        train=True,
        transform=T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize((0.5), (1))
        ]),
        download=True
    )
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True
    )
else:
    raise Exception("Not a valid dataset")

G = Generator(args.num_noises, NUM_COLORS, args.depths, IMAGE_SIZE)
D = Discriminator(NUM_COLORS, args.depths, IMAGE_SIZE)

criterion = torch.nn.BCELoss()
optimizer_g = torch.optim.Adam(
    G.parameters(),
    lr=args.learning_rate,
    betas=[args.beta_1, args.beta_2]
)
optimizer_d = torch.optim.Adam(
    D.parameters(),
    lr=args.learning_rate,
    betas=[args.beta_1, args.beta_2]
)

if __name__ == "__main__":
    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader):
            # Train D with genuine data
            genuine = data[0] # Drop label data
            genuine = genuine.reshape(-1, NUM_COLORS, IMAGE_SIZE, IMAGE_SIZE)

            optimizer_d.zero_grad()

            output = D(genuine)
            loss_d = criterion(output, torch.ones(output.shape))
            loss_d.backward()

            # Train D with fake data
            noise = torch.FloatTensor(args.num_noises).uniform_(-1, 1)
            fake = G(noise)

            output = D(fake.detach())
            loss_d = criterion(output, torch.zeros(output.shape))
            loss_d.backward()

            optimizer_d.step()

            # Train G with fake data
            optimizer_g.zero_grad()

            output = D(fake)
            loss_g = criterion(output, torch.ones(output.shape))
            loss_g.backward()

            optimizer_g.step()

    torch.save(G.state_dict(), "./Models/Gen-%d.pt" % args.epochs)
