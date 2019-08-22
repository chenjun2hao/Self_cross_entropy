import torch
import torch.nn.functional as F
import numpy as np 

# test1：正常情况下loss的计算
log_soft = torch.tensor([[-8.4646, -8.4646, -8.4646, -8.4646, -8.4646],
                         [-8.4646, -8.4646, -8.4646, -8.4646, -8.4646]])
target = torch.tensor([3, 1])

def Self_cross_entropy(input, target, ignore_index=None):
    '''自己用pytorch实现cross_entropy，
       有时候会因为各种原因，如：样本问题等，出现个别样本的loss为nan，影响模型的训练，
       不适用于所有样本loss都为nan的情况
       input:n*categ
       target:n
    '''
    input = input.contiguous().view(-1, input.shape[-1])
    log_prb = F.log_softmax(input, dim=1)

    one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)     # 将target转换成one-hot编码
    loss = -(one_hot * log_prb).sum(dim=1)                                  # n,得到每个样本的loss

    if ignore_index:                            # 忽略[PAD]的label
        non_pad_mask = target.ne(0)
        loss = loss.masked_select(non_pad_mask)
    
    not_nan_mask = ~torch.isnan(loss)           # 找到loss为非nan的样本
    loss = loss.masked_select(not_nan_mask).mean()
    return loss

print('正常计算loss的输出为：')
loss = Self_cross_entropy(log_soft, target)
print('自己写的loss：', loss)

entropy_out = F.cross_entropy(log_soft, target)
print('官方loss：', entropy_out)


# test2：有异常样本的loss计算
print('存在个别异常样本的输出为:')
''' log_soft中有个样本的输出全为nan，这就是一个错误样本，应该剔除掉， '''
content = torch.load('./state.pth')
log_soft = content['log_soft']
target = content['target']

loss = Self_cross_entropy(log_soft, target)
print('自己写的loss：', loss)

entropy_out = F.cross_entropy(log_soft, target)
print('官方loss：', entropy_out)



