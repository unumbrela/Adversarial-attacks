import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import torchattacks
from sklearn.metrics import accuracy_score
from PIL import Image

# 自定义数据集以区分 real 和 fake
class RealFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        self.transform = transform

        self.samples = []
        self.real_count = 0
        self.fake_count = 0

        # 加载 real 样本
        if os.path.isdir(self.real_dir):
            real_classes = sorted(os.listdir(self.real_dir))
            for cls in real_classes:
                cls_path = os.path.join(self.real_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, 0))  # 0 表示 real
                        self.real_count += 1
        else:
            print(f"Directory {self.real_dir} does not exist.")

        # 加载 fake 样本
        if os.path.isdir(self.fake_dir):
            fake_classes = sorted(os.listdir(self.fake_dir))
            for cls in fake_classes:
                cls_path = os.path.join(self.fake_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, 1))  # 1 表示 fake
                        self.fake_count += 1
        else:
            print(f"Directory {self.fake_dir} does not exist.")

        print(f"Total real images: {self.real_count}")
        print(f"Total fake images: {self.fake_count}")
        print(f"Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))  # 返回一张黑色图片
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(model, train_loader, criterion, optimizer, num_epochs=20, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total * 100
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    print('训练完成')

def evaluate(model, data_loader, attack=None, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.enable_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if attack:
                images.requires_grad = True
                images = attack(images, labels)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc

def evaluate_real_fake(model, data_loader, attack=None, device='cpu'):
    model.eval()
    real_preds = []
    real_labels = []
    fake_preds = []
    fake_labels = []
    with torch.enable_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if attack:
                images.requires_grad = True
                images = attack(images, labels)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            for p, l in zip(preds, labels):
                if l == 0:
                    real_preds.append(p)
                    real_labels.append(l)
                else:
                    fake_preds.append(p)
                    fake_labels.append(l)
    real_acc = accuracy_score(real_labels, real_preds) if real_labels else 0
    fake_acc = accuracy_score(fake_labels, fake_preds) if fake_labels else 0
    return real_acc, fake_acc

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 数据预处理，添加数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                             std=[0.229, 0.224, 0.225])   # ImageNet 标准差
    ])

    # 加载数据
    dataset = RealFakeDataset(root_dir='../data', transform=transform)
    print(f'数据集总大小: {len(dataset)}')

    # 数据集划分
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}')

    # 设置 DataLoader
    # 将 num_workers 设置为 0 以避免在 Windows 上的多进程死锁问题
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    # 使用预训练的 ResNet18 模型
    model = models.resnet18(pretrained=True)

    # 修改最后的全连接层为二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 类别：real vs fake

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=20, device=device)

    # 保存模型（仅保存 state_dict）
    torch.save(model.state_dict(), 'model.pth')
    print('模型已保存为 model.pth')

    # 对抗攻击评估
    # 加载训练好的模型
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model = model.to(device)

    # 定义对抗攻击（FGSM）
    attack = torchattacks.FGSM(model, eps=0.007)

    # 评估在原始数据上的准确率
    original_acc = evaluate(model, test_loader, device=device)
    print(f'原始数据上的总体准确率: {original_acc * 100:.2f}%')

    # 评估在对抗攻击下的准确率
    adversarial_acc = evaluate(model, test_loader, attack=attack, device=device)
    print(f'对抗攻击后的总体准确率: {adversarial_acc * 100:.2f}%')

    # 分别评估 real 和 fake 的准确率
    # 评估原始数据
    original_real_acc, original_fake_acc = evaluate_real_fake(model, test_loader, device=device)
    print(f'原始数据 - 真实图片准确率: {original_real_acc * 100:.2f}%, 假图片准确率: {original_fake_acc * 100:.2f}%')

    # 评估对抗攻击后的数据
    adversarial_real_acc, adversarial_fake_acc = evaluate_real_fake(model, test_loader, attack=attack, device=device)
    print(f'对抗攻击后 - 真实图片准确率: {adversarial_real_acc * 100:.2f}%, 假图片准确率: {adversarial_fake_acc * 100:.2f}%')

if __name__ == '__main__':
    main()
