"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import nice
from nice import LogisticPriorNICELoss
import matplotlib.pyplot as plt
import os


def train(flow, trainloader, optimizer, epoch, prior, device, log_pt, model_save_filename):

    flow.train()  # set to training mode

    # === choose which loss function to build:
    if prior == 'logistic':
        nice_loss_fn = LogisticPriorNICELoss(size_average=True)
    else:
        v = 0

        # TODO >> Correct to some massage about irrelevant...

    # def loss_fn(fx):
    #     """Compute NICE loss w/r/t a prior and optional L1 regularization."""
    #     return nice_loss_fn(fx, flow.scaling_diag)
    #
    #     # if args.lmbda == 0.0:
    #     #     return nice_loss_fn(fx, flow.scaling_diag)
    #     # else:
    #     #     return nice_loss_fn(fx, flow.scaling_diag) + args.lmbda*l1_norm(model, include_bias=True)

    running_loss = 0.0

    for inputs, _ in tqdm(trainloader):
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[
            3])  # change  shape from BxCxHxW to Bx(C*H*W)
        #loss = -(flow(inputs.to(device))).mean()
        loss = -flow(inputs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        running_loss += loss.item()

    if epoch % log_pt == 0:
        torch.save(flow.state_dict(), "./models/" + model_save_filename)
        print(f"Epoch {epoch}:  train loss: {running_loss / len(trainloader)} ")

    return running_loss / len(trainloader)


def test(flow, testloader, epoch, prior, device, model_save_filename, sample_shape):
    flow.train()  # set to training mode

    # === choose which loss function to build:
    if prior == 'logistic':
        nice_loss_fn = LogisticPriorNICELoss(size_average=True)
    else:
        v = 0

        # TODO >> Correct to some massage about irrelevant...

    # def loss_fn(fx):
    #     """Compute NICE loss w/r/t a prior and optional L1 regularization."""
    #     return nice_loss_fn(fx, flow.scaling_diag)
    #
    #     # if args.lmbda == 0.0:
    #     #     return nice_loss_fn(fx, flow.scaling_diag)
    #     # else:
    #     #     return nice_loss_fn(fx, flow.scaling_diag) + args.lmbda*l1_norm(model, include_bias=True)

    flow.eval()  # set to inference mode
    with torch.no_grad():
        samples = flow.sample(100).cpu()
        noise = torch.distributions.Uniform(0., 1.).sample(samples.size())
        samples = (samples * 255. + noise) / 256.
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + model_save_filename + 'epoch%d.png' % epoch)
        # TODO full in

        running_loss = 0.0

        for inputs, _ in tqdm(testloader):
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[
                3])  # change  shape from BxCxHxW to Bx(C*H*W)
            #loss = -(flow(inputs.to(device))).mean()
            loss = -(flow(inputs).mean()
            running_loss += loss.item()
        return running_loss / len(testloader)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    full_dim = 28 * 28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%s_' % args.batch_size \
                          + 'coupling%s_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%s_' % args.mid_dim \
                          + 'hidden%s_' % args.hidden \
                          + '.pt'

    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=full_dim,
        hidden_dim=args.mid_dim,
        hidden_layers=args.hidden,
        device=device).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    # TODO fill in

    train_loss = []
    test_loss = []

    for epoch in range(args.epochs):
        train_loss.append(train(flow, trainloader, optimizer, epoch, args.prior, device,args.log_pt,model_save_filename))
        test_loss.append(test(flow, testloader, epoch, args.prior, device, model_save_filename, sample_shape))



    # Save plot of loss values :

    fig, ax = plt.subplots()
    ax.plot(train_loss)
    ax.plot(test_loss)
    ax.set_title("Train & Test Log Likelihood Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Train Loss", "Test Loss"])
    plt.savefig(os.path.join(os.getcwd(), "loss.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=32)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--log_pt',
                        help='save model cpt every X epochs',
                        type=int,
                        default=3)

    args = parser.parse_args()
    main(args)
