import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 优先选择gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# LeNet 5用于对Mnist数据集分类
class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        # 5*5卷积核卷积
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        # 3*3卷积核卷积（为了输入28*28能够匹配，原网络是5*5卷积核）
        self.conv2 = nn.Conv2d(6, 16, 3)  # 卷积核为方阵的时候可以只传入一个参数
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        前向传播函数，返回为一个size为[batch_size,features]的向量
        :param x:
        :return:
        """
        # 卷积+relu+池化
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

    net = CnnNet().to(device)
    # 准则函数使用交叉熵函数，可以尝试其他
    criterion = nn.CrossEntropyLoss()
    # 优化方法为带动量的随机梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(20):
        print('Start training: epoch {}'.format(epoch+1))
        train(train_loader)
        test(test_loader)
