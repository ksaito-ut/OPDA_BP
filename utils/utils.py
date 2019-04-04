import torch.optim as opt
from models.basenet import *
import torch

def get_model(net, num_class=13, unit_size=100):
    if net == 'alex':
        model_g = AlexBase()
        model_c = Classifier(num_classes=num_class)
    elif net == 'vgg':
        model_g = VGGBase()
        model_c = Classifier(num_classes=num_class)
    else:
        model_g = ResBase(net, unit_size=unit_size)
        model_c = ResClassifier(num_classes=num_class, unit_size=unit_size)
    return model_g, model_c


def get_optimizer_visda(lr, G, C, update_lower=False):
    if not update_lower:
        params = list(list(G.linear1.parameters()) + list(G.linear2.parameters()) + list(
            G.bn1.parameters()) + list(G.bn2.parameters())) #+ list(G.bn4.parameters()) + list(
            #G.bn3.parameters()) + list(G.linear3.parameters()) + list(G.linear4.parameters()))
    else:
        params = G.parameters()
    optimizer_g = opt.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005,nesterov=True)
    optimizer_c = opt.SGD(list(C.parameters()), momentum=0.9, lr=lr,
                          weight_decay=0.0005, nesterov=True)
    return optimizer_g, optimizer_c


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c


def adjust_learning_rate(optimizer, lr, batch_id, max_id, epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    beta = 0.75
    alpha = 10
    p = min(1, (batch_id + max_id * epoch) / float(max_id * max_epoch))
    lr = lr / (1 + alpha * p) ** (beta)  # min(1, 2 - epoch/float(20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
