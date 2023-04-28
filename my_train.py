import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fasternet import *# 导入我们自己定义的模型文件

# 准备数据集
train_data = torchvision.datasets.CIFAR10("E:/learn_pytorch/src/dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("E:/learn_pytorch/src/dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

#length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度：{}, 测试数据集的长度为: {}".format(train_data_size, test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
tudui = FasterNet()

#创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
#learning_rate = 0.01
learning_rate = 1e-2  # 1e-2 = 1x(10)^(-2)=1/100 =0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的次数
epoch = 1

#添加tensorboard
writer = SummaryWriter("./logs/train")

for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))

    # 训练步骤开始
    tudui.train() # 这一步，表示网络的训练状态。如果我们的网络模型中有drop,batchnorm层则需要有这句话
    for data in train_dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0 :
            print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试
    tudui.eval() # 表示网络现在是测试模式
    total_test_loss = 0
    #整体正确的个数
    total_accuracy = 0
    with torch.no_grad(): # 不要梯度，保证不要调优
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size)
    total_test_step += 1

    # 模型保存
    torch.save(tudui, "tudui_CIFAR10_{}.pth".format(i))
    # 保存方式2： torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
