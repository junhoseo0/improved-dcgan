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
args = parser.parse_args()

if args.dataset == "mnist":
    from data_mnist import data, dataloader, IMAGE_SIZE
else:
    raise Exception("Not a valid dataset")

G = Generator(args.num_noises, args.colors, args.depths, IMAGE_SIZE)
D = Discriminator(args.colors, args.depths, IMAGE_SIZE)

criterion = torch.nn.CrossEntropyLoss()
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
