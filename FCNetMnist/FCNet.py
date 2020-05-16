import torch
import torch.nn as nn
from torch.optim import optimizer
from torchvision import datasets, transforms

# 优先选择gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # 一共三层神经网络，一个隐层
        self.features = nn.Sequential(
            nn.Linear(784, 100),
            nn.Sigmoid(),
            nn.Linear(100, 10)
        )

    # 前向传播
    def forward(self, x):
        # 输入为16*1*28*28，这里转换为16*784
        x = x.view(16, -1)
        output = self.features(x)
        return output


# 训练网络
def train(train_loader):
    # 损失函数值
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # 如果有gpu，则使用gpu
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度置零
        optimizer.zero_grad()
        # 前向传播
        output = net(inputs)
        # 损失函数
        loss = criterion(output, labels)
        # 反向传播，权值更新
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 每50个batch_size后打印一次损失函数值
        if i % 100 == 99:
            print('%5d loss: %.3f' %
                  (i + 1, running_loss / 100))
            running_loss = 0.0


# 训练完1个或几个epoch之后，在测试集上测试以下准确率，防止过拟合
def test(test_loader):
    correct = 0
    total = 0
    # 不进行autograd
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
            100 * correct / total))
    return correct / total


if __name__ == '__main__':
    # Pytorch自带Mnist数据集，可以直接下载，分为测试集和训练集
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    # DataLoader类可以实现数据集的分批和打乱等
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    net = FCNet().to(device)
    # 准则函数使用交叉熵函数，可以尝试其他
    criterion = nn.CrossEntropyLoss()
    # 优化方法为带动量的随机梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        print('Start training: epoch {}'.format(epoch+1))
        train(train_loader)
        test(test_loader)