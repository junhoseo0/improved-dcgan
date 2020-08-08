import argparse
import os.path as path
import torch.cuda
import torchvision
import torchvision.transforms as T
from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", type=int, default=10)
parser.add_argument("--num-noises", "-nn", type=int, default=100)
parser.add_argument("--num-colors", "-nc", type=int, default=3)
parser.add_argument("--depths", "-d", type=int, default=128)
parser.add_argument("--image-size", "-is", type=int, default=64)
args = parser.parse_args()

if args.image_size % 16 != 0:
    raise Exception("Size of the image must be divisible by 16")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/g%d.pt" % args.epochs
OUTPUT_PATH = "./images/"
NUM_IMAGES = 10

G = Generator(args.num_noises, args.num_colors, args.depths, args.image_size)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

for i in range(NUM_IMAGES):
    noise = torch.FloatTensor(args.num_noises).uniform_(-1, 1)
    fake = G(noise)
    torchvision.utils.save_image(
        fake.view(args.num_colors, args.image_size, args.image_size),
        path.join(OUTPUT_PATH, "g%d_image%d.bmp" % (args.epochs, i))
    )
