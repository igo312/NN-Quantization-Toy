# train a resnet18 in cifar10
import pdb

import torchvision
import torch
from torchvision import transforms
import os
from tqdm import tqdm

def test(model, loader, device="cuda"):
    total, correct = 0, 0
    with torch.no_grad():
        print('testing...')
        for data in tqdm(loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    return 100 * correct // total

if __name__ == '__main__':
    ###### data prepare ##############
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((128,128)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ########## train ###########
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, len(classes))
    epochs = 50
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    train_interval = 50
    ckpt_path = './ckpt/'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    accuracy = 0
    device = 'cuda'

    ### training
    model = model.to(device)
    # only save best
    for epoch in range(epochs):
        for iter, data in enumerate(trainloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = lossFunction(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter%train_interval == 0:
                print("[Training Epoch:{} Iter:{}/{} ] Loss is {}".format(epoch, iter, len(trainloader), loss))
        acc = test(model, testloader)
        if acc>accuracy:
            print('[Testing Epoch:{}] the best acc is {}, save to {}'.format(epoch, acc, ckpt_path))
            torch.save(dict(
                state_dict = model.state_dict(),
            ), os.path.join(ckpt_path, 'mobilenet_cifar10_best.pth'))
            accuracy = acc
