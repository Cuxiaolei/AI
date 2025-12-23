# -*- coding:utf-8 -*-

import torch
import os
import numpy as np
from Model_Block import Resnet1d, Resnet2d
from Model_Fusion_Network import MahFusion_Network
from Model_Fusion_Loss import Fusion_loss
from Data_Sampler import TrainSampler, TestSampler
from Parser import Parser
import glob

def Split_model_param(model):
    model_params = []
    ft_params = []
    for name, param in model.named_parameters():
        name = name.split('.')
        if name[-1] == 'gamma' or name[-1] == 'beta':
            ft_params.append(param)
        else:
            model_params.append(param)

    return model_params, ft_params

def Train(opt, model, result_path):
    Tsampler = TrainSampler(opt=opt)
    scaler = Tsampler.scaler
    Titer = iter(Tsampler)
    loss_fn = Fusion_loss(opt=opt)
    model_params, ft_params = Split_model_param(model=model)
    model_optim = torch.optim.Adam(model_params, lr=opt.lr)
    ft_optim = torch.optim.Adam(ft_params, weight_decay=1e-8, lr=opt.ft_lr)
    max_acc = 0
    for epoch in range(1, 1+opt.epochs):
        ps_loss, ps_acc = 0, 0
        pu_loss, pu_acc = 0, 0
        for episode in range(1, 1+opt.episodes):
            ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq, pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq = next(Titer)
            for weight in model_params:
                weight.fast = None

            model.train()
            psTFq_output, psTFs_proto, psTF_variance, psDEq_output, psDEs_proto, psDE_variance, psFFTq_output, psFFTs_proto, psFFT_variance = model.forward(TFs=ps_TFs, TFq=ps_TFq,
                                                                                                                                                            DEs=ps_DEs, DEq=ps_DEq,
                                                                                                                                                            FFTs=ps_FFTs, FFTq=ps_FFTq)
            psloss, psacc = loss_fn.forward(TFq_output=psTFq_output, TFs_proto=psTFs_proto, TF_variance=psTF_variance, DEq_output=psDEq_output, DEs_proto=psDEs_proto,
                                        DE_variance=psDE_variance, FFTq_output=psFFTq_output, FFTs_proto=psFFTs_proto, FFT_variance=psFFT_variance)

            meta_grad = torch.autograd.grad(psloss, model_params, create_graph=True)
            for k, weight in enumerate(Split_model_param(model=model)[0]):
                weight.fast = weight - opt.lr * meta_grad[k]
            meta_grad = [g.detach() for g in meta_grad]

            model.eval()
            puTFq_output, puTFs_proto, puTF_variance, puDEq_output, puDEs_proto, puDE_variance, puFFTq_output, puFFTs_proto, puFFT_variance = model.forward(TFs=pu_TFs, TFq=pu_TFq,
                                                                                                                                                            DEs=pu_DEs, DEq=pu_DEq,
                                                                                                                                                            FFTs=pu_FFTs, FFTq=pu_FFTq)
            puloss, puacc = loss_fn.forward(TFq_output=puTFq_output, TFs_proto=puTFs_proto, TF_variance=puTF_variance, DEq_output=puDEq_output, DEs_proto=puDEs_proto,
                                            DE_variance=puDE_variance, FFTq_output=puFFTq_output, FFTs_proto=puFFTs_proto, FFT_variance=puFFT_variance)

            model_optim.zero_grad()
            for k, weight in enumerate(Split_model_param(model=model)[0]):
                weight.grad = meta_grad[k]
            model_optim.step()

            ft_optim.zero_grad()
            puloss1 = puloss.detach_().requires_grad_(True)
            puloss1.backward()
            ft_optim.step()

            ps_loss += psloss.item()
            ps_acc += psacc.item()
            pu_loss += puloss.item()
            pu_acc += puacc.item()
        ps_loss = ps_loss / opt.episodes
        ps_acc = ps_acc / opt.episodes
        pu_loss = pu_loss / opt.episodes
        pu_acc = pu_acc / opt.episodes
        print('=======In Epoch {}=======, model loss is {:6f}, model accuracy is {:4f}, ft loss is {:6f}, ft accuracy is {:4f}'.format(epoch, ps_loss, ps_acc, pu_loss, pu_acc))
        if epoch % 10 == 0:
            model_path = os.path.normpath(os.path.join(result_path, str(epoch)+'.pth'))
            torch.save(model.state_dict(), model_path)
    return scaler

def Test(opt, model, scaler):
    Vsampler = TestSampler(opt=opt,scaler=scaler)
    Viter = iter(Vsampler)
    test_acc = 0
    print('{}'.format('='*30))
    print('{}'.format('='*30))
    print('======Test with the last model======')
    loss_fn = Fusion_loss(opt=opt)
    result = []
    model.eval()
    with torch.no_grad():
        model_params, _ = Split_model_param(model=model)
        for weight in model_params:
            weight.fast = None
        for i in range(1, 501):
            TFs, DEs, FFTs, TFq, DEq, FFTq = next(Viter)
            TFq_output, TFs_proto, TF_variance, DEq_output, DEs_proto, DE_variance, FFTq_output, FFTs_proto, FFT_variance = model.forward(TFs=TFs, TFq=TFq,
                                                                                                                                          DEs=DEs, DEq=DEq,
                                                                                                                                          FFTs=FFTs, FFTq=FFTq)
            loss, acc = loss_fn.forward(TFq_output=TFq_output, TFs_proto=TFs_proto, TF_variance=TF_variance, DEq_output=DEq_output,
                                        DEs_proto=DEs_proto, DE_variance=DE_variance, FFTq_output=FFTq_output, FFTs_proto=FFTs_proto, FFT_variance=FFT_variance)
            test_acc += acc.item()
            avg_acc = test_acc / i
            if i % 100 == 0:
                result.append(avg_acc)
                print('=======Number of iteration is {}, test accuracy is {:4f}======='.format(i, avg_acc))
    return result

def main():
    opt = Parser().parse_args()
    for i in range(3):
        per_expername = str(opt.k_val)+'way'+str(opt.n_val)+'shot-'+'['+opt.train_domain+'——'+opt.test_domain+']'+'-Exp'+str(i)
        per_result_path = os.path.join(opt.base_dir, per_expername)
        if not os.path.exists(per_result_path):
            os.makedirs(per_result_path)
        trainmodel = MahFusion_Network(TF_encoder=Resnet2d(blockdim=[64, 64, 128, 256, 512],Feature_trans=True),
                                       DE_encoder=Resnet1d(blockdim=[32, 32, 64, 128, 256], Feature_trans=True),
                                       FFT_encoder=Resnet1d(blockdim=[16, 16, 32, 64, 128], Feature_trans=True)).cuda()
        scaler = Train(opt=opt, model=trainmodel, result_path=per_result_path)
        testmodel = MahFusion_Network(TF_encoder=Resnet2d(blockdim=[64, 64, 128, 256, 512],Feature_trans=True),
                                       DE_encoder=Resnet1d(blockdim=[32, 32, 64, 128, 256], Feature_trans=True),
                                       FFT_encoder=Resnet1d(blockdim=[16, 16, 32, 64, 128], Feature_trans=True)).cuda()
        model_path = os.path.join(per_result_path, str(opt.epochs)+'.pth')
        testmodel.load_state_dict(torch.load(model_path), strict=False)
        result = Test(opt=opt, model=testmodel, scaler=scaler)
        np.savetxt(per_result_path+'/result——'+str(i)+'.csv', np.array(result), fmt='%.6f')












