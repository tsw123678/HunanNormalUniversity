from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import pickle
from data_set import channel_set_gen

def complex_mul(h, x): # h fixed on batch, x has multiple batch
    if len(h.shape) == 1:
        # h is same over all messages (if estimated h, it is averaged)
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[0] - x[:, 1] * h[1]
        y[:, 1] = x[:, 0] * h[1] + x[:, 1] * h[0]
    elif len(h.shape) == 2:
        # h_estimated is not averaged
        assert x.shape[0] == h.shape[0]
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[:, 0] - x[:, 1] * h[:, 1]
        y[:, 1] = x[:, 0] * h[:, 1] + x[:, 1] * h[:, 0]
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError       #尚未实现的方法
    return y

def complex_mul_taps(h, x_tensor):
    if len(h.shape) == 1:
        L = h.shape[0] // 2  # length/2 of channel vector means number of taps  6/2=3,确实是tap的个数
    elif len(h.shape) == 2:
        L = h.shape[1] // 2  # length/2 of channel vector means number of taps
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    y = torch.zeros(x_tensor.shape[0], x_tensor.shape[1], dtype=torch.float)
    assert x_tensor.shape[1] % 2 == 0
    for ind_channel_use in range(x_tensor.shape[1]//2):     # "//" : 取整除 - 返回商的整数部分（向下取整）
        for ind_conv in range(min(L, ind_channel_use+1)):
            if len(h.shape) == 1:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(h[2*ind_conv:2*(ind_conv+1)], x_tensor[:, (ind_channel_use-ind_conv)*2:(ind_channel_use-ind_conv+1)*2])
            else:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(
                    h[:, 2 * ind_conv:2 * (ind_conv + 1)],
                    x_tensor[:, (ind_channel_use - ind_conv) * 2:(ind_channel_use - ind_conv + 1) * 2])

    return y

def complex_conv_transpose(h_trans, y_tensor): # takes the role of inverse filtering  起到反滤波的作用？？？
    assert len(y_tensor.shape) == 2 # batch
    assert y_tensor.shape[1] % 2 == 0
    assert h_trans.shape[0] % 2 == 0
    if len(h_trans.shape) == 1:
        L = h_trans.shape[0] // 2
    elif len(h_trans.shape) == 2:
        L = h_trans.shape[1] // 2
    else:
        print('h shape length need to be either 1 or 2')

    deconv_y = torch.zeros(y_tensor.shape[0], y_tensor.shape[1] + 2*(L-1), dtype=torch.float)
    for ind_y in range(y_tensor.shape[1]//2):
        ind_y_deconv = ind_y + (L-1)
        for ind_conv in range(L):
            if len(h_trans.shape) == 1:
                deconv_y[:, 2*(ind_y_deconv - ind_conv):2*(ind_y_deconv - ind_conv+1)] += complex_mul(h_trans[2*ind_conv:2*(ind_conv+1)] , y_tensor[:,2*ind_y:2*(ind_y+1)])
            else:
                deconv_y[:, 2 * (ind_y_deconv - ind_conv):2 * (ind_y_deconv - ind_conv + 1)] += complex_mul(
                    h_trans[:, 2 * ind_conv:2 * (ind_conv + 1)], y_tensor[:, 2 * ind_y:2 * (ind_y + 1)])
    return deconv_y[:, 2*(L-1):]

# 产生h
# f_meta_channels = open("training_channels.pckl", 'rb')      #以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。一般用于非文本文件如图片等。
# h_list = pickle.load(f_meta_channels)          #把f_meta_channels中的值赋给h_list_meta
# f_meta_channels.close()
# print(h_list)

h_list = channel_set_gen(1, 1, True)
# h_list = channel_set_gen(1, 3, False)


##自编码机  详见P7第二段的描述
class basic_DNN(nn.Module):
    def __init__(self, M=1024, num_neurons_encoder=512, n=128, n_inv_filter=2,
                 num_neurons_decoder=512, if_bias=True, if_relu=True, if_RTN=False, snr=25,rali=True):
        super(basic_DNN, self).__init__()
        self.enc_fc1 = nn.Linear(M, num_neurons_encoder, bias=if_bias)
        self.enc_fc2 = nn.Linear(num_neurons_encoder, n, bias=if_bias)

        ### norm, nothing to train
        ### channel, nothing to train

        num_inv_filter = 2 * n_inv_filter       #P8最后一行末尾
        if if_RTN:
            self.rtn_1 = nn.Linear(n, n, bias=if_bias)
            self.rtn_2 = nn.Linear(n, n, bias=if_bias)
            self.rtn_3 = nn.Linear(n, num_inv_filter, bias=if_bias)
        else:
            pass

        self.dec_fc1 = nn.Linear(n, num_neurons_decoder, bias=if_bias)
        self.dec_fc2 = nn.Linear(num_neurons_decoder, M, bias=if_bias)
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(M)

        # 产生噪音
        Eb_over_N = pow(10, (snr / 10))
        R = 1
        noise_var = 1 / (2 * R * Eb_over_N)  # 不知道这三行是什么意思
        self.Noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(128),
                                                                           noise_var * torch.eye(128))

        self.rali = rali

    def forward(self, x, h=h_list[0], device="cuda", if_RTN=False):
        h.to(device)
        x = self.enc_fc1(x)
        x = self.activ(x)
        x = self.enc_fc2(x)

        if self.rali:
            # normalize
            x_norm = torch.norm(x, dim=1)       #指定dim = 1，即去掉其dim=1，所以是横向求值，计算Frobenius范数  返回给定张量的矩阵范数或向量范数  计算指定维度的范数

            x_norm = x_norm.unsqueeze(1)        #Returns a new tensor with a dimension of size one inserted at the specified position.

            x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm  # since each has ^2 norm as 0.5 -> complex 1

            # channel
            x = complex_mul_taps(h, x)          #P3第三段最后 where。。。
            x = x.to(device)

        # noise
        n = torch.zeros(x.shape[0], x.shape[1])
        for noise_batch_ind in range(x.shape[0]):
            n[noise_batch_ind] = self.Noise.sample()        #sample()作用：和channel_set_gen中else下类似
        n = n.type(torch.FloatTensor).to(device)
        x = x + n # noise insertion

        # RTN
        if if_RTN:
            h_inv = self.rtn_1(x)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_2(h_inv)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_3(h_inv) # no activation for the final rtn (linear activation without weights)
            x = complex_conv_transpose(h_inv, x)
            x = x.to(device)
        else:
            pass
        x = self.dec_fc1(x)
        x = self.activ(x)
        x = self.dec_fc2(x) # softmax taken at loss function
        x = self.norm(x)
        return x


## DSN网络
class DSN(nn.Module):
    def __init__(self,):
        super(DSN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1,bias=True),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1,bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,4,padding=1,bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.output = nn.Conv2d(512,1,4,padding=1,bias=True)

    def forward(self,x):
        x = self.backbone(x)
        x = self.output(x)
        return x

##自编码机  详见P7第二段的描述
# class basic_DNN(nn.Module):
#     def __init__(self, M, num_neurons_encoder, n, n_inv_filter, num_neurons_decoder, if_bias, if_relu, if_RTN):
#         super(basic_DNN, self).__init__()
#         self.enc_fc1 = nn.Linear(M, num_neurons_encoder, bias=if_bias)
#         #self.enc_fc2 = nn.Linear(num_neurons_encoder, n, bias=if_bias)
#         self.enc_conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7)
#         ### norm, nothing to train
#         ### channel, nothing to train
#
#         num_inv_filter = 2 * n_inv_filter       #P8最后一行末尾
#         if if_RTN:
#             self.rtn_1 = nn.Linear(n, n, bias=if_bias)
#             self.rtn_2 = nn.Linear(n, n, bias=if_bias)
#             self.rtn_3 = nn.Linear(n, num_inv_filter, bias=if_bias)
#         else:
#             pass
#
#         self.dec_conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
#         #self.dec_fc1 = nn.Linear(n, num_neurons_decoder, bias=if_bias)
#         self.dec_fc2 = nn.Linear(num_neurons_decoder-7+1, M, bias=if_bias)
#         if if_relu:
#             self.activ = nn.ReLU()
#         else:
#             self.activ = nn.Tanh()
#         self.tanh = nn.Tanh()
#     def forward(self, x, h, noise_dist, device, if_RTN):
#         x = self.enc_fc1(x)
#         x = self.activ(x)
#         x = x.unsqueeze(1)
# #         x = x.permute(0,2,1)
#         #x = self.enc_fc2(x)
#         x = self.enc_conv2(x)
#         x = x.squeeze(1)
#         # normalize
#         x_norm = torch.norm(x, dim=1)       #指定dim = 1，即去掉其dim=1，所以是横向求值，计算Frobenius范数  返回给定张量的矩阵范数或向量范数  计算指定维度的范数
#         x_norm = x_norm.unsqueeze(1)        #Returns a new tensor with a dimension of size one inserted at the specified position.
#         x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm  # since each has ^2 norm as 0.5 -> complex 1
#         # channel
#         x = complex_mul_taps(h, x)          #P3第三段最后 where。。。
#         x = x.to(device)
#         #noise
# #         x = x.permute(1,0)
#         n = torch.zeros(x.shape[0], x.shape[1])
#         for noise_batch_ind in range(x.shape[0]):
#             n[noise_batch_ind] = noise_dist.sample()        #sample()作用：和channel_set_gen中else下类似
#         n = n.type(torch.FloatTensor).to(device)
#         x = x + n # noise insertion
# #         x = x.permute(1,0)
#         # RTN
#         if if_RTN:
#             h_inv = self.rtn_1(x)
#             h_inv = self.tanh(h_inv)
#             h_inv = self.rtn_2(h_inv)
#             h_inv = self.tanh(h_inv)
#             h_inv = self.rtn_3(h_inv) # no activation for the final rtn (linear activation without weights)
# #             x = h_inv
#             x = complex_conv_transpose(h_inv, x)
#             x = x.to(device)
#         else:
#             pass
#
#         x = x.unsqueeze(1)
#         x = self.dec_conv1(x)
# #         x = x.permute(0,2,1)
#         x = x.squeeze(1)
#         x = self.activ(x)
#         x = self.dec_fc2(x) # softmax taken at loss function
#         return x


def dnn(**kwargs):          #两个星号**的作用是把dict类型的数据作为参数传入。
    net = basic_DNN(**kwargs)
    return net

from torchsummary import summary

if __name__ == '__main__':

    net = DSN()

    net.to("cuda")
    summary(net,(3,224,224))
