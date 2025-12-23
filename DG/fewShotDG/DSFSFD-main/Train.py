# -*- coding:utf-8 -*-  # 指定文件编码为UTF-8，支持中文注释

# 导入必要的库
import torch  # PyTorch深度学习框架
import os  # 操作系统相关操作（路径、文件夹创建等）
import numpy as np  # 数值计算库
from Model_Block import Resnet1d, Resnet2d  # 导入1D和2D ResNet编码器（自定义模块）
from Model_Fusion_Network import MahFusion_Network  # 导入多模态融合网络（自定义模块）
from Model_Fusion_Loss import Fusion_loss  # 导入融合损失函数（自定义模块）
from Data_Sampler import TrainSampler, TestSampler  # 导入训练/测试数据采样器（自定义模块）
from Parser import Parser  # 导入参数解析器（自定义模块，用于解析命令行参数）
import glob  # 文件路径匹配（此处未直接使用，可能用于后续扩展）


def Split_model_param(model):
    """
    分割模型参数为两类：普通模型参数和批归一化层参数（gamma/beta）
    目的：对不同类型参数使用不同的优化策略（批归一化参数通常需要更精细的调优）
    Args:
        model: 待分割参数的模型
    Returns:
        model_params: 普通模型参数（非gamma/beta）
        ft_params: 批归一化层参数（gamma/beta，通常用于fine-tune）
    """
    model_params = []  # 存储普通模型参数
    ft_params = []  # 存储批归一化层参数（gamma/beta）
    for name, param in model.named_parameters():  # 遍历模型所有参数及其名称
        name = name.split('.')  # 分割参数名称（例如"bn1.gamma"分割为["bn1", "gamma"]）
        # 如果参数是批归一化的gamma（缩放系数）或beta（偏移系数），归入ft_params
        if name[-1] == 'gamma' or name[-1] == 'beta':
            ft_params.append(param)
        else:
            model_params.append(param)  # 其他参数归入model_params
    return model_params, ft_params


def Train(opt, model, result_path):
    """
    模型训练函数（核心），采用元学习的"内循环-外循环"训练策略
    Args:
        opt: 命令行参数（包含训练轮次、学习率等配置）
        model: 待训练的模型（MahFusion_Network）
        result_path: 模型保存路径
    Returns:
        scaler: 数据标准化器（用于测试时保持数据分布一致）
    """
    # 初始化训练数据采样器（生成episodic数据，包含支持集和查询集）
    Tsampler = TrainSampler(opt=opt)
    scaler = Tsampler.scaler  # 获取采样器中的数据标准化器（用于测试时复用）
    Titer = iter(Tsampler)  # 创建采样器的迭代器，用于批量获取数据

    # 初始化损失函数（融合损失，针对多模态数据设计）
    loss_fn = Fusion_loss(opt=opt)

    # 分割模型参数为普通参数和批归一化参数
    model_params, ft_params = Split_model_param(model=model)

    # 初始化优化器：分别优化普通参数和批归一化参数
    model_optim = torch.optim.Adam(model_params, lr=opt.lr)  # 普通参数优化器（学习率为opt.lr）
    ft_optim = torch.optim.Adam(ft_params, weight_decay=1e-8, lr=opt.ft_lr)  # 批归一化参数优化器（更小的学习率和权重衰减）

    max_acc = 0  # 记录最大准确率（此处未实际使用，可用于早停）

    # 训练主循环（按epoch迭代）
    for epoch in range(1, 1 + opt.epochs):  # 从1到opt.epochs（总训练轮次）
        # 初始化损失和准确率累加器（ps: support set 支持集；pu: query set 查询集）
        ps_loss, ps_acc = 0, 0  # 支持集损失和准确率
        pu_loss, pu_acc = 0, 0  # 查询集损失和准确率

        # 每个epoch包含多个episode（元学习的核心：每个episode模拟一个"任务"）
        for episode in range(1, 1 + opt.episodes):
            # 从采样器获取当前episode的数据（多模态数据：TF/DE/FFT，分为支持集ps和查询集pu）
            # ps_TFs: 支持集的TF特征；ps_TFq: 支持集的查询样本TF特征（此处命名可能有误，实际应为ps是支持集，pu是查询集）
            ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq, pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq = next(
                Titer)

            # 重置模型参数的fast属性（元学习中用于内循环快速更新的临时参数）
            for weight in model_params:
                weight.fast = None  # fast参数存储内循环更新后的权重

            # 内循环：在支持集上训练，更新模型（快速适应）
            model.train()  # 模型切换到训练模式
            # 支持集前向传播：输入支持集的多模态特征，输出查询样本的预测、原型和方差
            (psTFq_output, psTFs_proto, psTF_variance,  # TF模态输出
            psDEq_output, psDEs_proto, psDE_variance,  # DE模态输出
            psFFTq_output, psFFTs_proto, psFFT_variance) = model.forward(  # FFT模态输出
                TFs=ps_TFs, TFq=ps_TFq,
                DEs=ps_DEs, DEq=ps_DEq,
                FFTs=ps_FFTs, FFTq=ps_FFTq
            )
            # 计算支持集的损失和准确率
            psloss, psacc = loss_fn.forward(
                TFq_output=psTFq_output, TFs_proto=psTFs_proto, TF_variance=psTF_variance,
                DEq_output=psDEq_output, DEs_proto=psDEs_proto, DE_variance=psDE_variance,
                FFTq_output=psFFTq_output, FFTs_proto=psFFTs_proto, FFT_variance=psFFT_variance
            )

            # 计算支持集损失关于模型参数的梯度（内循环梯度，用于快速更新）
            meta_grad = torch.autograd.grad(psloss, model_params, create_graph=True)  # create_graph=True保留计算图，用于外循环梯度计算
            # 内循环参数更新：用支持集梯度更新模型参数（存储在fast属性中，不改变原始参数）
            for k, weight in enumerate(Split_model_param(model=model)[0]):
                weight.fast = weight - opt.lr * meta_grad[k]  # 梯度下降更新（学习率为opt.lr）
            #  detach元梯度（避免后续计算影响内循环）
            meta_grad = [g.detach() for g in meta_grad]

            # 外循环：在查询集上评估，更新原始模型参数（元优化）
            model.eval()  # 模型切换到评估模式（避免批归一化等层的训练行为）
            # 查询集前向传播：使用内循环更新后的fast参数进行预测
            (puTFq_output, puTFs_proto, puTF_variance,
            puDEq_output, puDEs_proto, puDE_variance,
            puFFTq_output, puFFTs_proto, puFFT_variance) = model.forward(
                TFs=pu_TFs, TFq=pu_TFq,
                DEs=pu_DEs, DEq=pu_DEq,
                FFTs=pu_FFTs, FFTq=pu_FFTq
            )
            # 计算查询集的损失和准确率
            puloss, puacc = loss_fn.forward(
                TFq_output=puTFq_output, TFs_proto=puTFs_proto, TF_variance=puTF_variance,
                DEq_output=puDEq_output, DEs_proto=puDEs_proto, DE_variance=puDE_variance,
                FFTq_output=puFFTq_output, FFTs_proto=puFFTs_proto, FFT_variance=puFFT_variance
            )

            # 外循环更新普通模型参数（使用内循环得到的meta_grad）
            model_optim.zero_grad()  # 清空梯度
            for k, weight in enumerate(Split_model_param(model=model)[0]):
                weight.grad = meta_grad[k]  # 赋值元梯度
            model_optim.step()  # 执行优化步骤

            # 更新批归一化参数（使用查询集损失）
            ft_optim.zero_grad()  # 清空梯度

            ft_optim.zero_grad()
            puloss.backward()  # 不要 detach
            ft_optim.step()

            # 累加当前episode的损失和准确率
            ps_loss += psloss.item()  # 支持集损失累加
            ps_acc += psacc.item()  # 支持集准确率累加
            pu_loss += puloss.item()  # 查询集损失累加
            pu_acc += puacc.item()  # 查询集准确率累加

        # 计算当前epoch的平均损失和准确率（除以episode数量）
        ps_loss = ps_loss / opt.episodes
        ps_acc = ps_acc / opt.episodes
        pu_loss = pu_loss / opt.episodes
        pu_acc = pu_acc / opt.episodes

        # 打印当前epoch的训练结果
        print(
            '=======In Epoch {}=======, model loss is {:6f}, model accuracy is {:4f}, ft loss is {:6f}, ft accuracy is {:4f}'.format(
                epoch, ps_loss, ps_acc, pu_loss, pu_acc
            ))

        # 每10个epoch保存一次模型
        if epoch % 10 == 0:
            model_path = os.path.normpath(os.path.join(result_path, str(epoch) + '.pth'))  # 模型保存路径
            torch.save(model.state_dict(), model_path)  # 保存模型参数

    return scaler  # 返回数据标准化器（供测试使用）


def Test(opt, model, scaler):
    """
    模型测试函数：用训练好的模型在测试集上评估性能
    Args:
        opt: 命令行参数
        model: 训练好的模型
        scaler: 训练时使用的数据标准化器（保证测试数据分布一致）
    Returns:
        result: 测试过程中的准确率列表
    """
    # 初始化测试数据采样器（使用训练时的scaler标准化数据）
    Vsampler = TestSampler(opt=opt, scaler=scaler)
    Viter = iter(Vsampler)  # 创建测试数据迭代器
    test_acc = 0  # 测试准确率累加器
    print('{}'.format('=' * 30))
    print('{}'.format('=' * 30))
    print('======Test with the last model======')  # 提示：使用最后一个epoch的模型进行测试

    loss_fn = Fusion_loss(opt=opt)  # 初始化损失函数（测试时主要用其计算准确率）
    result = []  # 存储测试过程中的平均准确率
    model.eval()  # 模型切换到评估模式

    with torch.no_grad():  # 关闭梯度计算（测试阶段不更新参数，节省计算资源）
        model_params, _ = Split_model_param(model=model)  # 获取模型参数
        for weight in model_params:
            weight.fast = None  # 重置fast参数（测试时不进行内循环更新）

        # 测试循环（共500次迭代）
        for i in range(1, 501):
            # 获取测试数据（多模态特征）
            TFs, DEs, FFTs, TFq, DEq, FFTq = next(Viter)

            # 模型前向传播（测试数据）
            (TFq_output, TFs_proto, TF_variance,
            DEq_output, DEs_proto, DE_variance,
            FFTq_output, FFTs_proto, FFT_variance) = model.forward(
                TFs=TFs, TFq=TFq,
                DEs=DEs, DEq=DEq,
                FFTs=FFTs, FFTq=FFTq
            )

            # 计算测试损失和准确率（主要关注准确率）
            loss, acc = loss_fn.forward(
                TFq_output=TFq_output, TFs_proto=TFs_proto, TF_variance=TF_variance,
                DEq_output=DEq_output, DEs_proto=DEs_proto, DE_variance=DE_variance,
                FFTq_output=FFTq_output, FFTs_proto=FFTs_proto, FFT_variance=FFT_variance
            )

            # 累加准确率并计算平均
            test_acc += acc.item()
            avg_acc = test_acc / i  # 截至当前迭代的平均准确率

            # 每100次迭代记录一次平均准确率并打印
            if i % 100 == 0:
                result.append(avg_acc)
                print('=======Number of iteration is {}, test accuracy is {:4f}======='.format(i, avg_acc))

    return result  # 返回测试准确率列表


def main():
    """主函数：串联训练和测试流程，控制实验循环"""
    opt = Parser().parse_args()  # 解析命令行参数（如训练轮次、学习率、任务设置等）

    # 重复3次实验（为了验证结果的稳定性）
    for i in range(3):
        # 构建当前实验的名称和结果保存路径
        per_expername = f"{opt.k_val}way{opt.n_val}shot-[{opt.train_domain}——{opt.test_domain}]-Exp{i}"  # 实验名称（包含任务设置、域信息、实验序号）
        per_result_path = os.path.join(opt.base_dir, per_expername)  # 结果保存路径
        if not os.path.exists(per_result_path):  # 若路径不存在则创建
            os.makedirs(per_result_path)

        # 初始化训练模型：多模态融合网络（MahFusion_Network）
        # 包含3个编码器：TF用2D ResNet，DE和FFT用1D ResNet（Feature_trans=True表示需要特征转换）
        trainmodel = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=[64, 64, 128, 256, 512], Feature_trans=True),
            DE_encoder=Resnet1d(blockdim=[32, 32, 64, 128, 256], Feature_trans=True),
            FFT_encoder=Resnet1d(blockdim=[16, 16, 32, 64, 128], Feature_trans=True)
        ).cuda()  # 将模型放到GPU上

        # 训练模型并获取数据标准化器
        scaler = Train(opt=opt, model=trainmodel, result_path=per_result_path)

        # 初始化测试模型（结构与训练模型一致）
        testmodel = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=[64, 64, 128, 256, 512], Feature_trans=True),
            DE_encoder=Resnet1d(blockdim=[32, 32, 64, 128, 256], Feature_trans=True),
            FFT_encoder=Resnet1d(blockdim=[16, 16, 32, 64, 128], Feature_trans=True)
        ).cuda()

        # 加载训练好的模型参数（最后一个epoch的模型）
        model_path = os.path.join(per_result_path, f"{opt.epochs}.pth")
        testmodel.load_state_dict(torch.load(model_path), strict=False)  # strict=False允许参数不严格匹配（如测试时无需训练相关参数）

        # 测试模型并获取结果
        result = Test(opt=opt, model=testmodel, scaler=scaler)

        # 将测试结果保存为CSV文件
        np.savetxt(f"{per_result_path}/result——{i}.csv", np.array(result), fmt='%.6f')


# 当脚本直接运行时，执行主函数
if __name__ == "__main__":
    main()