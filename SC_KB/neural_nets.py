
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import numpy as np
from torch.autograd import Variable

# new comment
class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64 * 64 * 3),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# maybe it can be smaller
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self,in_dim=3,SAC=False):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, 12, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.use_SAC = SAC
        self.Mask = MaskNet(48)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, in_dim, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, latent = None):
        if latent == None:
            self.mu, self.logvar = self.encode(x)
            z = self.reparametrize(self.mu, self.logvar)
            self.z_shape = z.shape
            if self.use_SAC:
                z = self.Mask(z)
            latent = z.view(z.size(0), -1)
            return latent
        else:
            z = latent.view(self.z_shape)
            z = self.decode(z)
        return z, self.mu, self.logvar

def loss_vae(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = F.mse_loss(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

class mlp(nn.Module):
    def __init__(self, in_size=28*28*1, num_classes=10):
        super(mlp, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        print(x.shape)
        out = x.view(x.size(0), -1)

        out = self.linear(out)
        return out

class lstm(nn.Module):
    def __init__(self, input_size=32, hidden_size=128, num_layers=2, num_classes=10):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.reshape(-1, self.input_size, self.input_size)
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
# 2
class CAE(nn.Module):
    def __init__(self,input_dim=3, SAC=False):
        super(CAE, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=5,bias=False)
        # return_indices 返回每个最大值的索引位置
        self.pool = nn.MaxPool2d((2, 2),return_indices=True)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5,bias=False)
        self.use_SAC = SAC
        self.Mask = MaskNet(32)
        self.convt1= nn.ConvTranspose2d(32, 128, kernel_size=5)
        self.convt2 = nn.ConvTranspose2d(128, input_dim, kernel_size=5)
        self.uppool = nn.MaxUnpool2d(2, 2)

    def forward(self, x, latent = None):
        if latent == None:
            # Leaky ReLU 允许一小部分的负值通过，而不是将所有负值都置为零
            x = F.leaky_relu(self.conv1(x))
            x, self.indices1 = self.pool(x)
            x = F.leaky_relu(self.conv2(x))
            x, self.indices2 = self.pool(x)
            self.x_shape = x.shape

            # 提取语义像素
            if self.use_SAC:
                x = self.Mask(x)
            latent = x.view(x.size(0), -1)
            return latent
        else:
            x = latent.view(self.x_shape)
            x = self.uppool(x,self.indices2)
            x = F.leaky_relu(self.convt1(x))
            x = self.uppool(x,self.indices1)
            # -1~1
            x = F.tanh(self.convt2(x))
            return x

class CNN(nn.Module):
    def __init__(self,input_dim=3):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.cls = nn.Linear(512, 10) # mnist:512, cifar10:800

    def forward(self, x):
        x = self.encoder(x)
        latent = x.view(x.size(0), -1)
        cls = self.cls(latent)
        return latent, cls

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, size=512, out=3):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg11s():
    return VGG(make_layers([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M']), size=128)

def vgg11():
    return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']))
# 2.5
class MaskNet(nn.Module):
    def __init__(self,input_dim=32):
        super(MaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(128, input_dim, kernel_size=3,padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        mask = self.conv2(y)
        mask = torch.sign(mask)
        mask = F.relu(mask)
        x = torch.mul(x, mask)
        print(x.shape)
        index = torch.where(x!=0)
        retain_x = x[index]
        print("压缩语义bit:", retain_x.element_size() * retain_x.nelement())

        return x

if __name__ == '__main__':
    # net = VAE()

    # net=CAE()
    net=MaskNet()
    net.to("cuda")
    # summary(net,(3,64,64),device="cuda")
    summary(net,(32,13,13),device="cuda")

