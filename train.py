import argparse
import torch
from dcgan import Generator, Discriminator

parser = argparse.ArgumentParser("Hyperparameters and Dataset Arguments")
parser.add_argument("--dataset", "-ds", type=str, default="mnist")
parser.add_argument("--num-noises", "-n", type=int, default=100)
parser.add_argument("--colors", "-c", type=int, default=3)
parser.add_argument("--depths", "-d", type=int, default=128)
parser.add_argument("--learning-rate", "-lr", type=float, default=0.0002)
parser.add_argument("--beta-1", "-b1", type=float, default=0.5)
parser.add_argument("--beta-2", "-b2", type=float, default=0.99)
parser.add_argument("--epochs", "-e", type=int, default=10)
args = parser.parse_args()

if args.dataset == "mnist":
    from data_mnist import data, dataloader, IMAGE_SIZE
else:
    raise Exception("Not a valid dataset")

G = Generator(args.num_noises, args.colors, args.depths, IMAGE_SIZE)
D = Discriminator(args.colors, args.depths, IMAGE_SIZE)

criterion = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(
    G.parameters(),
    lr=args.learning_rate,
    betas=[args.beta_1, args.beta_2]
)
optimizer_D = torch.optim.Adam(
    D.parameters(),
    lr=args.learning_rate,
    betas=[args.beta_1, args.beta_2]
)

if __name__ == "__main__":
    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader):
            # Train D with genuine data
            genuine = data[0] # Drop label data
            genuine = genuine.reshape(-1, args.colors, IMAGE_SIZE, IMAGE_SIZE)

            optimizer_D.zero_grad()

            output = D(genuine)
            loss_d = criterion(output, torch.ones(output.shape))
            loss_d.backward()

            # Train D with fake data
            noise = torch.FloatTensor(args.num_noises).uniform_(-1, 1)
            fake = G(noise)

            output = D(fake.detach())
            loss_d = criterion(output, torch.zeros(output.shape))
            loss_d.backward()

            optimizer_D.step()

            # Train G with fake data
            optimizer_G.zero_grad()

            output = D(fake)
            loss_g = criterion(output, torch.ones(output.shape))
            loss_g.backward()

            optimizer_D.step()
    
    torch.save(G.state_dict(), "./Models/Gen-%d.pt" % args.epochs)
