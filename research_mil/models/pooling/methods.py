import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BagClassifier(nn.Module):
    def __init__(self, in_channels, classes, embed_size=512):
        super().__init__()

        self.embedder   = nn.Sequential(nn.Linear(in_channels, embed_size))
        self.classifier = nn.Linear(embed_size, classes)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels)


class bagAverage(BagClassifier):
    def __init__(self, in_channels, classes, embed):
        super(bagAverage, self).__init__(in_channels, classes, embed)

    def forward(self, x, norm=True):

        out_feat = self.embedder(x)
        out = self.classifier(out_feat)
        if norm:
            return out, F.normalize(out_feat)
        else:
            return out, out_feat


class BagNetwork(nn.Module):

    def __init__(self, backbone, pooling, classes=2, embed_size=512, k=32):
        super().__init__()

        self.features = backbone
        self.pooling  = pooling
        self.mode     = 0
        channels      = 1
        k             = k

        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=channels, kernel_size=1, stride=1),
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=channels, kernel_size=1, stride=1),
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=channels, kernel_size=1, stride=1),
        )

        # what if a bag has less than k
        # we should select one random feature map and expand it to match k
        # 16x16 is too small
        # lets push it to 32x32 -> 1024

        self.bag_module = nn.Sequential(
            nn.Conv2d(in_channels=k * channels, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=channels+1, kernel_size=1, stride=1),
        )

        self.bag_classifier = nn.Sequential(
            nn.Linear(embed_size, classes),
        )

        for i in [self.bag_module, self.p4,self.p3,self.p2]:
            for m in i:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        if self.mode == 0: # instance mode
            x = self.features(x)[0]
            x, feat = self.pooling(x)
            return x, feat

        else: # combined mode

            bsize = x.shape[0]
            x     = x.view((-1, 3, 256, 256))

            x, out  = self.features(x)
            x, feat = self.pooling(x)

            out_x4 = self.p4(out['x4']) # 8
            out_x3 = self.p3(out['x3']) # 16
            out_x2 = self.p2(out['x2']) # 32

            out_y = F.interpolate(out_x4.clone(), scale_factor=2) + out_x3
            out_y = F.interpolate(out_y.clone(),  scale_factor=2) + out_x2
            out_y = out_y.squeeze() # 32x32

            out_y = out_y.view((bsize, -1, 32, 32)) # N,K,16,16

            # This is a bit of a hack for bags with less than 32 instances
            if out_y.shape[1] < 32:
                buff  = torch.zeros(bsize, (32 - out_y.shape[1]), 32, 32).float().to(x.device)
                out_y = torch.cat([out_y, buff], 1)

            out_bag = self.bag_module(out_y) # N,channels+1,16,16
            # N,channels+1,16,16 -> 512
            out_bag_embed = self.pooling.embedder(out_bag.view(out_bag.shape[0],-1))

            # normalize for centerloss
            out_bag = self.bag_classifier(out_bag_embed)

            return x, feat, F.normalize(out_bag_embed), out_bag

    def setmode(self, mode):
        self.mode = mode





if __name__ == '__main__':
    import sys, os 
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

    from models.backbones import load_backbone
    
    # input bag [1,32,3,256,256] , N = 1, K = 32 , C=3, H = 256, W = 256
    x  = torch.randn(2 , 32, 3, 256, 256)

    backbone = load_backbone('resnet34v2',True)
    pooling  = bagAverage(in_channels=512, classes=2, embed=512)
    #print(backbone)

    net = BagNetwork(backbone, pooling, classes=2, embed_size=512, k=32)
    net.mode = 1
    print(net)
    print()

    print('Input - Instance Predication - Instance Features - Bag Normalized Features - Bag Prediction')
    print(x.shape, ' -> ', net(x)[0].shape,' ', net(x)[1].shape,' ', net(x)[2].shape,' ', net(x)[3].shape, '\n', sep='')
