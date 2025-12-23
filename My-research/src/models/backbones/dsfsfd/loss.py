# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def Mahalanobis_dist(xq, proto, variance):          #xq: (k*q, dim), proto: (k, dim), variance: (k, 1)
    num = xq.size(0)
    dim = xq.size(1)
    protonum = proto.size(0)
    xq = xq.unsqueeze(dim=1).expand(num, protonum, dim)
    proto = proto.unsqueeze(dim=0).expand(num, protonum, dim)
    variance = variance.view(1, protonum).expand(num, protonum)
    Mah_dist = (pow(xq-proto, 2).sum(dim=2))/variance                   #Tensor(k*q, k)

    return Mah_dist

class Fusion_loss(nn.Module):
    def __init__(self, opt):
        super(Fusion_loss, self).__init__()
        self.TF_weight = opt.TF_weight
        self.DE_weight = opt.DE_weight
        self.FFT_weight = opt.FFT_weight

    def forward(self, TFq_output, TFs_proto, TF_variance, DEq_output, DEs_proto, DE_variance, FFTq_output, FFTs_proto, FFT_variance):
        #Setting true label
        k = TFq_output.size(0)
        q = TFq_output.size(1)

        y = torch.from_numpy(np.repeat(range(k), q)).view(-1, 1).to(torch.long).cuda()
        #Time Frequency Loss
        TFq_output = TFq_output.view(k*q, -1)
        TF_Mahdist = Mahalanobis_dist(xq=TFq_output, proto=TFs_proto, variance=TF_variance)
        TF_pred = self.TF_weight * F.softmax(-TF_Mahdist, dim=1)                                                        #Tensor(k*q, k)
        TF_Mahloss = -(self.TF_weight * F.log_softmax(-TF_Mahdist, dim=1))                                              #Tensor(k*q, k)

        #DE Loss
        DEq_output = DEq_output.view(k*q, -1)
        DE_Mahdist = Mahalanobis_dist(xq=DEq_output, proto=DEs_proto, variance=DE_variance)
        DE_pred = self.DE_weight * F.softmax(-DE_Mahdist, dim=1)                                                        #Tensor(k*q, k)
        DE_Mahloss = -(self.DE_weight * F.log_softmax(-DE_Mahdist, dim=1))                                              #Tensor(k*q, k)

        #FFT Loss
        FFTq_output = FFTq_output.view(k*q, -1)
        FFT_Mahdist = Mahalanobis_dist(xq=FFTq_output, proto=FFTs_proto, variance=FFT_variance)
        FFT_pred = self.FFT_weight * F.softmax(-FFT_Mahdist, dim=1)                                                     #Tensor(k*q, k)
        FFT_Mahloss = -(self.FFT_weight * F.log_softmax(-FFT_Mahdist, dim=1))                                           #Tensor(k*q, k)

        #Three Modal Fusion loss
        Fusion_loss = TF_Mahloss+DE_Mahloss+FFT_Mahloss
        loss = torch.gather(Fusion_loss, dim=1, index=y).squeeze().mean()
        pred_Mah = TF_pred + DE_pred + FFT_pred
        _, pred = torch.max(pred_Mah, dim=1)
        acc = torch.eq(y.view(-1), pred).to(torch.float).mean()

        return loss, acc