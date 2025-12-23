# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv_block2d(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
def Conv_block1d(in_channel, out_channel):
    return nn.Sequential(nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
                          nn.BatchNorm1d(out_channel), nn.ReLU(inplace=True))

class MahFusion_Network(nn.Module):
    def __init__(self, TF_encoder, DE_encoder, FFT_encoder):
        super(MahFusion_Network, self).__init__()
        self.TF_encoder = TF_encoder
        self.TF_variance = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, 64), nn.ReLU(inplace=True),
                                         nn.Linear(64, 1), nn.ReLU(inplace=True))

        self.DE_encoder = DE_encoder
        self.DE_variance = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True),
                                         nn.Linear(256, 256), nn.ReLU(inplace=True),
                                         nn.Linear(256, 64), nn.ReLU(inplace=True),
                                         nn.Linear(64, 1), nn.ReLU(inplace=True))

        self.FFT_encoder = FFT_encoder
        self.FFT_variance = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True),
                                          nn.Linear(128, 128), nn.ReLU(inplace=True),
                                          nn.Linear(128, 64), nn.ReLU(inplace=True),
                                          nn.Linear(64, 1), nn.ReLU(inplace=True))

    def forward(self, TFs, TFq, DEs, DEq, FFTs, FFTq):
        #Time Frequency forward
        k = TFs.size(0)
        n = TFs.size(1)
        q = TFq.size(1)

        TFs, TFq = TFs.view(k*n, 3, 224, 224), TFq.view(k*q, 3, 224, 224)
        TFs_output = self.TF_encoder.forward(TFs).view(k, n, -1)
        TFq_output = self.TF_encoder.forward(TFq).view(k, q, -1)                                                        #Tensor(k, q, 512)
        TFs_proto = torch.mean(TFs_output, dim=1)                                                                       #Tensor(k, 512)
        #TFs_protovar = TFs_proto.view(k, 512, 1, 1)
        TF_variance = F.softplus(self.TF_variance(TFs_proto)).squeeze().view(k, 1)                                      #Tensor(k, 1)

        #DE forward

        DEs, DEq = DEs.view(k*n, 1, 512), DEq.view(k*q, 1, 512)
        DEs_output = self.DE_encoder.forward(DEs).view(k, n, -1)
        DEq_output = self.DE_encoder.forward(DEq).view(k, q, -1)                                                        #Tensor(k, q, 256)
        DEs_proto = torch.mean(DEs_output, dim=1)                                                                       #Tensor(k, 256)
        #DEs_protovar = DEs_proto.view(k, 256, 1)
        DE_variance = F.softplus(self.DE_variance(DEs_proto)).squeeze().view(k, 1)                                      #Tensor(k, 1)

        #FFT forward
        FFTs, FFTq = FFTs.view(k*n, 1, 256), FFTq.view(k*q, 1, 256)
        FFTs_output = self.FFT_encoder.forward(FFTs).view(k, n, -1)
        FFTq_output = self.FFT_encoder.forward(FFTq).view(k, q, -1)                                                     #Tensor(k, q, 128)
        FFTs_proto = torch.mean(FFTs_output, dim=1)                                                                     #Tensor(k, 128)
        #FFTs_protovar = FFTs_proto.view(k, 128, 1)
        FFT_variance = F.softplus(self.FFT_variance(FFTs_proto)).squeeze().view(k, 1)                                   #Tensor(k, 1)

        return TFq_output, TFs_proto, TF_variance, DEq_output, DEs_proto, DE_variance, FFTq_output, FFTs_proto, FFT_variance