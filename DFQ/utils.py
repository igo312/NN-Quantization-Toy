import torchvision
import torch
from tqdm import tqdm

def test(model, loader, prefix='Model', device="cuda"):
    model = model.to(device)
    model.eval()

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
    print(f'[{prefix}]: Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def load_model(model, ckpt='./ckpt/mobilenet_cifar10_best.pth'):
    file = torch.load(ckpt)
    model.load_state_dict(file['state_dict'])
    return model