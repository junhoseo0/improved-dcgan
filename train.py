import argparse

parser = argparse.ArgumentParser("Hyperparameters and Dataset Arguments")
parser.add_argument("--dataset", "-d", type=str, default="mnist")
args = parser.parse_args()

if args.dataset == "mnist":
    from data_mnist import data, dataloader
else:
    raise Exception("Not a valid dataset")
