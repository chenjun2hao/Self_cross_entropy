**问题：**用pytorch的crossentropy做损失函数的时候，迭代几步之后loss为nan的情况，
调试了很久才发现是用attention模块的时候，mask全为0了，导致atten的值全为nan。于是自己写了一个剔除个别异常样本的cross_entropy

详情请看代码，测试输出为:
```python
正常计算loss的输出为：
自己写的loss： tensor(1.6094)
官方loss： tensor(1.6094)

存在个别异常样本的输出为:
自己写的loss： tensor(7.9369, device='cuda:0', grad_fn=<MeanBackward1>)
官方loss： tensor(nan, device='cuda:0', grad_fn=<NllLossBackward>)
```

**test data download links [state.pth](https://pan.baidu.com/s/1cvRMLEBdFR82MO38hk2a8w), password:fscg**

---

**其他参考解决方案**
1. 在pred_x上加一个很小的量，如1e-10
```python
loss = crossentropy(out+1e-8, target)
```
2. 采用更小的学习率
3. 做梯度裁剪
The recommended thing to do when using ReLUs is to clip the gradient。
参考自[here](https://stats.stackexchange.com/questions/108381/how-to-avoid-nan-in-using-relu-cross-entropy)
4. 还可能是数据有问题
比如这位的.[链接](https://blog.csdn.net/ch07013224/article/details/80324373)

---

**[参考]**
- https://stats.stackexchange.com/questions/108381/how-to-avoid-nan-in-using-relu-cross-entropy