# model summaries
import torch
import torch.utils.model_zoo as model_zoo
from torchsummary import summary

class Model0(torch.nn.Module):

    def __init__(self,group_length,activation = torch.nn.ReLU()) :
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv3 = torch.nn.Conv2d(6,16,kernel_size=3,stride=1,padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dense5 = torch.nn.Linear(16*56*56,int(group_length/2))
        self.dense6 = torch.nn.Linear(int(group_length/2),group_length)
        self.activation = activation

    def forward(self,x) :
        z = self.conv1(x)
        z = self.pool2(z)
        z = self.conv3(z)
        z = self.pool4(z)
        z = z.view(z.size(0),-1)
        z = self.dense5(z)
        z = self.activation(z)
        z = self.dense6(z)

        return z


class Model1(torch.nn.Module):

    def __init__(self,group_length,activation = torch.nn.ReLU()) :
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,4,kernel_size=3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        self.conv3 = torch.nn.Conv2d(4,5,kernel_size=3,stride=1,padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        self.dense5 = torch.nn.Linear(5*14*14,int(group_length/2))
        self.dense6 = torch.nn.Linear(int(group_length/2),group_length)
        self.activation = activation

    def forward(self,x) :
        z = self.conv1(x)
        z = self.activation(z)
        z = self.pool2(z)
        z = self.conv3(z)
        z = self.activation(z)
        z = self.pool4(z)
        z = z.view(z.size(0),-1)
        z = self.dense5(z)
        z = self.activation(z)
        z = self.dense6(z)

        return z


class AlexNet(torch.nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,11,stride=4,padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Conv2d(64,192,5,stride=1,padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Conv2d(192,384,3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384,256,3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256,256,3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view([x.size()[0],256 * 6 * 6])
        x = self.classifier(x)
        return x

model_urls = {
'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def alexnet_classifier(num_classes):
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(256 * 6 * 6, int(num_classes/2) ),
        torch.nn.BatchNorm1d(int(num_classes/2)),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(int(num_classes/2), num_classes),
    )
    return classifier

def alexnet(num_classes, pretrained=False, **kwargs):
    """AlexNet model architecture from the "One weird trick..." paper.
    Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        for p in model.features.parameters():
            p.requires_grad=False
    classifier = alexnet_classifier(num_classes)
    model.classifier = classifier
    return model


if __name__ == '__main__':
    model0 = Model0(10)
    model1 = Model1(2)
    model3 = alexnet(2)

    print(summary(model0, (3, 224, 224)))
    print('\n')
    print(summary(model1, (3, 224, 224)))
    print('\n')
    print(summary(model3, (3, 224, 224)))
    print('\n')