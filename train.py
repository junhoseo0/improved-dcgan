import argparse
from dcgan import Generator, Discriminator

parser = argparse.ArgumentParser("Hyperparameters and Dataset Arguments")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--num-noises", "-n", type=int, default=100)
parser.add_argument("--colors", "-c", type=int, default=3)
parser.add_argument("--depths", "-d", type=int, default=128)
args = parser.parse_args()

if args.dataset == "mnist":
    from data_mnist import data, dataloader, IMAGE_SIZE
else:
    raise Exception("Not a valid dataset")

G = Generator(args.num_noises, args.colors, args.depths, IMAGE_SIZE)
D = Discriminator(args.colors, args.depths, IMAGE_SIZE)
