import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

device = 'cuda:0' 
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST('', is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=100, shuffle=True)

def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            # 将数据发送到GPU
            x = x.view(-1, 28 * 28).to(device)
            y = y.to(device)
            output = net(x)
            _, predicted = torch.max(output, 1)
            n_correct += (predicted == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total
# def evaluate(test_data, net):
#     n_correct = 0
#     n_total = 0
#     with torch.no_grad():
#         for (x, y) in test_data:
#             output = net.forward(x.view(-1, 28 * 28))
#             for i, output in enumerate(output):
#                 if torch.argmax(output) == y[i]:
#                     n_correct += 1
#                 n_total += 1
#     return n_correct / n_total
def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)
    print('initial accuracy', evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters())
    
    for epoch in range(100):
        for (x, y) in train_data:
            # 将数据发送到GPU
            x = x.view(-1, 28 * 28).to(device)
            y = y.to(device)
            net.zero_grad()
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print('epoch', epoch, 'accuracy', evaluate(test_data, net))
    torch.save(net,'./model.pth')

if __name__ == "__main__":
    main()
# def main():
#     train_data = get_data_loader(is_train=True)
#     test_data = get_data_loader(is_train=False)
#     net = Net().to(device)
#     print('initial accuracy', evaluate(test_data, net))
#     optimizer = torch.optim.Adam(net.parameters())
#     for epoch in range(3):
#         for (x, y) in train_data:
#             net.zero_grad()
#             output = net.forward(x.view(-1, 28 * 28))
#             loss = torch.nn.functional.nll_loss(output, y)
#             loss.backward()
#             optimizer.step()
#         print('epoch', epoch, 'accuracy', evaluate(test_data, net))

    # for (n, (x, _)) in enumerate(test_data):
    #     if n > 100:
    #          break
    #     predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
    #     plt.figure(n)
    #     plt.imshow(x[0].view(28, 28))
    #     plt.title('prediction:' + str(int(predict)))
    # plt.show()


if __name__ == "__main__":
    main() 