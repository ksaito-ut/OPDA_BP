from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class VGGBase(nn.Module):
    # Model VGG
    def __init__(self):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        print(model_ft)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        print(mod)
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        if target:
            return x
        else:
            return x
        return x


class AlexBase(nn.Module):
    def __init__(self):
        super(AlexBase, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = []
        print(model_ft)
        for i in xrange(18):
            if i < 13:
                mod.append(model_ft.features[i])
        mod_upper = list(model_ft.classifier.children())
        mod_upper.pop()
        # print(mod)
        self.upper = nn.Sequential(*mod_upper)
        self.lower = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False, feat_return=False):
        x = self.lower(x)
        x = x.view(x.size(0), 9216)
        x = self.upper(x)
        feat = x
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))))
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))))
        if feat_return:
            return feat
        if target:
            return x
        else:
            return x


class Classifier(nn.Module):
    def __init__(self, num_classes=12):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
        self.fc3 = nn.Linear(100, num_classes)  # nn.Linear(100, num_classes)

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
            feat = x
            x = self.fc3(x)
        else:
            feat = x
            x = self.fc3(x)
        if return_feat:
            return x, feat
        return x

class ResBase(nn.Module):
    def __init__(self, option='resnet18', pret=True, unit_size=100):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        # default unit size 100
        self.linear1 = nn.Linear(2048, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear3 = nn.Linear(unit_size, unit_size)
        self.bn3 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear4 = nn.Linear(unit_size, unit_size)
        self.bn4 = nn.BatchNorm1d(unit_size, affine=True)
    def forward(self, x,reverse=False):

        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        # best with dropout
        if reverse:
            x = x.detach()


        x = F.dropout(F.relu(self.bn1(self.linear1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.linear2(x))), training=self.training)
        #x = F.dropout(F.relu(self.bn3(self.linear3(x))), training=self.training)
        #x = F.dropout(F.relu(self.bn4(self.linear4(x))), training=self.training)
        #x = F.relu(self.bn1(self.linear1(x)))
        #x = F.relu(self.bn2(self.linear2(x)))
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, unit_size=1000):
        super(ResClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(unit_size, num_classes)
        )

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x
