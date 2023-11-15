import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

from my_models import ViT, CrossViT   # rename the skeleton file for your implementation / comment before testing for ResNet
from early_stopping import EarlyStopping

es = EarlyStopping(patience=10)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    # parser.add_argument('--model', type=str, default='r18', help='model to train (default: r18)')
    parser.add_argument('--model', type=str, default='cvit', help='model to train (default: vit)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)/len(output)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break
    return loss.item()


def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if set == 'Validation':
        es.check(val_acc=correct / len(test_loader.dataset))
    return test_loss


def run(args):
    # Download and load the training data
    # TODO: add augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(padding=4, size=32),
        # transforms.RandomRotation(degrees=90),
        # ImageNet mean/std values should also fit okayish for CIFAR
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # TODO: adjust folder
    dataset = datasets.CIFAR10('/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/Lucas/exercises/datasets/train', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Download and load the test data
    # TODO: adjust folder
    testset = datasets.CIFAR10('/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/Lucas/exercises/datasets/test', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        # model = models.resnet18(pretrained=False)
        # TODO: pretrained model
        model = models.resnet18(pretrained=True)
    elif args.model == "vit":
        model = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1,
                    emb_dropout = 0.1) 
    elif args.model == "cvit":
        model = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, 
                         lg_dim = 128, sm_patch_size = 8, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 128, 
                         sm_enc_dim_head = 64, lg_patch_size = 16, 
                         lg_enc_depth = 2, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 128, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

    # Define the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # TODO: add early stop and save model
    # load model
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        if es.early_stop is not True:
            # Training
            train_loss = train(model, trainloader, optimizer, criterion, device, epoch)
            train_losses.append(train_loss)

            # Validation
            val_loss = test(model, device, valloader, criterion, set="Validation")
            val_losses.append(val_loss)

    trained_epo = len(train_losses) + 1    
    # Plot up to the epoch where training was stopped
    plt.plot(range(1, trained_epo), train_losses, label='Train Loss')
    plt.plot(range(1, trained_epo), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses for CViT')
    plt.savefig('loss_plot.png')  # Save the plot
    plt.show()
    torch.save({'state_dict': model.state_dict()}, 'checkpoint_{:03d}.ckp'.format(trained_epo))
    test(model, device, testloader, criterion)


if __name__ == '__main__':
    args = parse_args()
    run(args)
