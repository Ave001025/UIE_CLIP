import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg19
from torchvision import transforms
import torch
import torch.nn.functional as F
import utils
from torchvision import models
from torchvision import transforms
def rank_loss(x1, x2):
    rank_loss = nn.MarginRankingLoss(margin=0.5).cuda()
    x1 = torch.clamp(x1, min=-5, max=5)
    x2 = torch.clamp(x2, min=-5, max=5)
    L_rank = rank_loss(x1, x2, torch.zeros_like(x1).cuda()+1.0)
    
    return L_rank

def ranker_loss(model, img):
    # with torch.no_grad():
    pre_input = utils.preprocessing(img)
    score = model(**pre_input)['final_result']
    loss = torch.mean(F.sigmoid(-score))

    return loss

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        # vgg = vgg16(pretrained=True).cuda()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        h1 = self.to_relu_1_2(x1)
        h1 = self.to_relu_2_2(h1)
        h1 = self.to_relu_3_3(h1)
        h1 = self.to_relu_4_3(h1)

        h2 = self.to_relu_1_2(x2)
        h2 = self.to_relu_2_2(h2)
        h2 = self.to_relu_3_3(h2)
        h2 = self.to_relu_4_3(h2)

        return torch.mean(torch.abs(h1 - h2))

class perception_loss_norm_vgg19(nn.Module):
    def __init__(self):
        super(perception_loss_norm_vgg19, self).__init__()
        # vgg = vgg16(pretrained=True).cuda()
        features = vgg19(pretrained=True).features
        self.to_relu_5_4 = features[:-1]
        self.requires_grad_(False)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        h1 = self.to_relu_5_4(x1)
        h2 = self.to_relu_5_4(x2)

        return torch.mean(torch.abs(h1 - h2))

class perception_loss_norm(perception_loss):
    def __init__(self):
        super().__init__()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        return super().forward(x1, x2)

def make_perception_loss(args):
    if args is None:
        return perception_loss()
    class_dict = {
        (True, 16): perception_loss_norm,
        (True, 19): perception_loss_norm_vgg19,
        (False, 16): perception_loss,
    }
    has_norm = args.get('norm', False)
    layers = args.get('layers', 16)
    return class_dict[(has_norm, layers)]()


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        #vgg_model = models.vgg19()
        #pre_file = torch.load('./vgg19-dcbb9e9d.pth')
        #vgg_model.load_state_dict(pre_file)
        #vgg_pretrained_features = vgg_model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach()) 
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap 

            loss += self.weights[i] * contrastive
        return loss



class C2R(nn.Module):
    def __init__(self, ablation=False):

        super(C2R, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n1, n2, n3, n4, n5, n6,inp, weight = False):
        a_vgg, p_vgg, n1_vgg, n2_vgg, n3_vgg, n4_vgg, n5_vgg, n6_vgg = self.vgg(a), self.vgg(p), self.vgg(n1), self.vgg(
            n2), self.vgg(n3), self.vgg(n4), self.vgg(n5), self.vgg(n6)
        inp_vgg = self.vgg(inp)
        n1_weight, n2_weight, n3_weight, n4_weight, n5_weight, n6_weight,inp_weight = weight
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an1 = self.l1(a_vgg[i], n1_vgg[i].detach())
                d_an2 = self.l1(a_vgg[i], n2_vgg[i].detach())
                d_an3 = self.l1(a_vgg[i], n3_vgg[i].detach())
                d_an4 = self.l1(a_vgg[i], n4_vgg[i].detach())
                d_an5 = self.l1(a_vgg[i], n5_vgg[i].detach())
                d_an6 = self.l1(a_vgg[i], n6_vgg[i].detach())
                d_inp = self.l1(a_vgg[i], inp_vgg[i].detach())
                        
                contrastive = d_ap / (
                        d_an1 * n1_weight + d_an2 * n2_weight + d_an3 * n3_weight + d_an4 * n4_weight + d_an5 * n5_weight + d_an6 * n6_weight + d_inp * inp_weight + 1e-7)                        
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


    

def clip_loss(model, a,p):
    #with torch.no_grad():
    score_a = model.forward_test(lq = a)['attributes']
    score_p = model.forward_test(lq = p)['attributes']
    #tmp = (100 - score_a * 100) - 0.975*(100 - score_p * 100) 
    tmp = (1 - score_a) - 0.975*(1 - score_p) 
    z = torch.tensor(0)
    loss = torch.mean(torch.max(z,tmp))
    return loss


