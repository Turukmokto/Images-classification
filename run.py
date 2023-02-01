import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
import torch
import torch.nn as nn

from model import MyClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def myResnet():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 131))
    return MyClassifier(model, device, "resnet")


def myVgg():
    model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 131),
    )
    return MyClassifier(model, device, "vgg")


def main():
    input_path = "./fruits-360/"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'Training':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
        'Test':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ]),
    }

    image_datasets = {
        'Training':
            datasets.ImageFolder(input_path + 'Training', data_transforms['Training']),
        'Test':
            datasets.ImageFolder(input_path + 'Test', data_transforms['Test'])
    }

    dataloaders = {
        'Training':
            torch.utils.data.DataLoader(image_datasets['Training'],
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=0),  # for Kaggle
        'Test':
            torch.utils.data.DataLoader(image_datasets['Test'],
                                        batch_size=32,
                                        shuffle=False,
                                        num_workers=0)  # for Kaggle
    }

    resnet = myResnet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.model.fc.parameters())
    trained_resnet = resnet.train_model(dataloaders, image_datasets, criterion, optimizer, 50)
    torch.save(trained_resnet.state_dict(), 'resnet_weights.h5')

    vgg = myVgg()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg.model.classifier.parameters())
    trained_vgg = vgg.train_model(dataloaders, image_datasets, criterion, optimizer, 50)
    torch.save(trained_vgg.state_dict(), 'vgg_weights.h5')


if __name__ == '__main__':
    main()
