import argparse
import pickle
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
parser.add_argument("--alpha", "-sl", type=float, default=0.9)
parser.add_argument("--beta-1", "-b1", type=float, default=0.5)
parser.add_argument("--beta-2", "-b2", type=float, default=0.99)
parser.add_argument("--epochs", "-e", type=int, default=10)
parser.add_argument("--report", "-r", type=int, default=100)
parser.add_argument("--workers", "-w", type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "mnist":
    IMAGE_SIZE = 32
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
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )
else:
    raise Exception("Not a valid dataset")

G = Generator(args.num_noises, NUM_COLORS, args.depths, IMAGE_SIZE).to(device)
D = Discriminator(NUM_COLORS, args.depths, IMAGE_SIZE).to(device)

def init_weight(model):
    classname = model.__class__.__name__
    if classname.find('conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0, 0.02)

G.apply(init_weight)
D.apply(init_weight)

criterion_d = torch.nn.BCELoss()
criterion_g = torch.nn.MSELoss()
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
    # One-sided label smoothing
    pos_labels = torch.full((args.batch_size, 1), args.alpha, device=device)
    neg_labels = torch.zeros((args.batch_size, 1), device=device)

    for epoch in range(args.epochs):
        losses_d, losses_g = [], []
        for i, data in enumerate(dataloader):
            # Train D with genuine data
            genuine = data[0].to(device) # Drop label data
            genuine = genuine.reshape(-1, NUM_COLORS, IMAGE_SIZE, IMAGE_SIZE)

            optimizer_d.zero_grad()

            output = D(genuine)
            loss_d_geunine = criterion_d(output, pos_labels)
            loss_d_geunine.backward()

            # Train D with fake data
            noise = torch.FloatTensor(args.batch_size, args.num_noises).uniform_(-1, 1).to(device)
            fake = G(noise)

            output = D(fake.detach())
            loss_d_fake = criterion_d(output, neg_labels)
            loss_d_fake.backward()

            optimizer_d.step()

            # Train G with fake data
            optimizer_g.zero_grad()

            # Feature matching
            target = D.conv[:-3](genuine)
            output = D.conv[:-3](fake)
            loss_g = criterion_g(output, target)
            loss_g.backward()

            optimizer_g.step()

            # Report & record loss
            if i % args.report == args.report-1:
                print("[%d/%d][%d/%d] D:%.7f G:%.7f" %
                    (epoch, args.epochs, i, len(dataloader), loss_d.item(), loss_g.item()))

            loss_d = loss_d_geunine + loss_d_fake
            losses_d.append(loss_d.item())
            losses_g.append(loss_g.item())

    # Save generator model
    torch.save(G.state_dict(), "./models/g%d.pt" % args.epochs)

    # Save loss records
    with open("./metrics/loss_g%d.pkl" % args.epochs, "wb") as g:
        pickle.dump(losses_g, g)
    with open("./metrics/loss_d%d.pkl" % args.epochs, "wb") as d:
        pickle.dump(losses_d, d)
