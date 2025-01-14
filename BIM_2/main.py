import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchattacks
from sklearn.metrics import accuracy_score
from PIL import Image
import logging
from torch.optim.lr_scheduler import StepLR

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_loading.log"),
        logging.StreamHandler()
    ]
)

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
            logging.info(f"Found {len(real_classes)} classes in real directory.")
            for cls in real_classes:
                cls_path = os.path.join(self.real_dir, cls)
                if not os.path.isdir(cls_path):
                    logging.warning(f"Skipping {cls_path} as it's not a directory.")
                    continue
                class_files = os.listdir(cls_path)
                logging.info(f"Loading images from class '{cls}' in real. Total images: {len(class_files)}")
                for img_name in class_files:
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        try:
                            with Image.open(img_path) as img:
                                img.verify()  # 验证图片是否损坏
                            self.samples.append((img_path, 0))  # 0 表示 real
                            self.real_count += 1
                        except Exception as e:
                            logging.error(f"Skipping corrupted image {img_path}: {e}")
        else:
            logging.error(f"Directory {self.real_dir} does not exist.")

        # 加载 fake 样本
        if os.path.isdir(self.fake_dir):
            fake_classes = sorted(os.listdir(self.fake_dir))
            logging.info(f"Found {len(fake_classes)} classes in fake directory.")
            for cls in fake_classes:
                cls_path = os.path.join(self.fake_dir, cls)
                if not os.path.isdir(cls_path):
                    logging.warning(f"Skipping {cls_path} as it's not a directory.")
                    continue
                class_files = os.listdir(cls_path)
                logging.info(f"Loading images from class '{cls}' in fake. Total images: {len(class_files)}")
                for img_name in class_files:
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        try:
                            with Image.open(img_path) as img:
                                img.verify()  # 验证图片是否损坏
                            self.samples.append((img_path, 1))  # 1 表示 fake
                            self.fake_count += 1
                        except Exception as e:
                            logging.error(f"Skipping corrupted image {img_path}: {e}")
        else:
            logging.error(f"Directory {self.fake_dir} does not exist.")

        logging.info(f"Total real images: {self.real_count}")
        logging.info(f"Total fake images: {self.fake_count}")
        logging.info(f"Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))  # 返回一张黑色图片
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, device='cpu'):
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

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total * 100
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    logging.info('训练完成')

def evaluate(model, data_loader, attack=None, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.enable_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 如果指定了攻击，则对当前batch的images进行扰动
            if attack:
                images = attack(images, labels)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc

def evaluate_real_fake(model, test_real_loader, test_fake_loader, attack_real=None, attack_fake=None, device='cpu'):
    # 评估真实图片
    if attack_real:
        real_acc = evaluate(model, test_real_loader, attack=attack_real, device=device)
    else:
        real_acc = evaluate(model, test_real_loader, attack=None, device=device)

    # 评估假图片
    if attack_fake:
        fake_acc = evaluate(model, test_fake_loader, attack=attack_fake, device=device)
    else:
        fake_acc = evaluate(model, test_fake_loader, attack=None, device=device)

    return real_acc, fake_acc

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'使用设备: {device}')

    # 数据预处理，添加数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                             std=[0.229, 0.224, 0.225])  # ImageNet标准差
    ])

    # 加载数据
    dataset = RealFakeDataset(root_dir='../data', transform=transform)
    logging.info(f'数据集总大小: {len(dataset)}')

    # 期望 real=4000, fake=4000 => 总共8000张
    expected_total = 4000 * 2
    if len(dataset) != expected_total:
        logging.warning(f"数据集大小为 {len(dataset)}，与预期的 {expected_total} 不符。请检查数据集是否完整。")

    # 数据集划分 (train / test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    logging.info(f'训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}')

    # 从 test_dataset 中单独取出 real / fake
    test_real_indices = [i for i, (img, label) in enumerate(test_dataset) if label == 0]
    test_fake_indices = [i for i, (img, label) in enumerate(test_dataset) if label == 1]

    test_real_dataset = Subset(test_dataset, test_real_indices)
    test_fake_dataset = Subset(test_dataset, test_fake_indices)

    logging.info(f'测试集中的真实图片数量: {len(test_real_dataset)}, 假图片数量: {len(test_fake_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    test_real_loader = DataLoader(test_real_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    test_fake_loader = DataLoader(test_fake_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

    # 使用预训练的 ResNet18 模型
    model = models.resnet18(pretrained=True)
    # 修改最后的全连接层为二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 类别：real vs fake
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 检查是否已经存在模型
    model_path = 'model.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info('成功加载已有的模型')
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            logging.info('开始重新训练模型')
            train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)
            torch.save(model.state_dict(), model_path)
            logging.info('模型已保存为 model.pth')
    else:
        logging.info('未找到模型，开始训练模型')
        train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)
        torch.save(model.state_dict(), model_path)
        logging.info('模型已保存为 model.pth')

    # -------------------------
    # 改动部分：使用 BIM 攻击
    # -------------------------
    # 与原本 PGD 攻击同参数 (eps=0.002, alpha=0.001, steps=10)
    # 你也可根据实验需要自行微调
    attack_real = torchattacks.BIM(model, eps=0.002, alpha=0.001, steps=10)
    attack_fake = torchattacks.BIM(model, eps=0.002, alpha=0.001, steps=10)

    # 评估对抗攻击前的准确率
    real_acc_pre, fake_acc_pre = evaluate_real_fake(
        model,
        test_real_loader,
        test_fake_loader,
        attack_real=None,
        attack_fake=None,
        device=device
    )
    logging.info(f'对抗攻击前 - 真实图片准确率: {real_acc_pre * 100:.2f}%, 假图片准确率: {fake_acc_pre * 100:.2f}%')

    # 评估对抗攻击后的准确率
    real_acc_post, fake_acc_post = evaluate_real_fake(
        model,
        test_real_loader,
        test_fake_loader,
        attack_real=attack_real,
        attack_fake=attack_fake,
        device=device
    )
    logging.info(f'对抗攻击后 - 真实图片准确率: {real_acc_post * 100:.2f}%, 假图片准确率: {fake_acc_post * 100:.2f}%')

    # 总体准确率（可选）
    total_test_real = len(test_real_dataset)
    total_test_fake = len(test_fake_dataset)
    total_test = total_test_real + total_test_fake
    overall_acc_pre = (real_acc_pre * total_test_real + fake_acc_pre * total_test_fake) / total_test
    overall_acc_post = (real_acc_post * total_test_real + fake_acc_post * total_test_fake) / total_test
    logging.info(f'对抗攻击前 - 总体准确率: {overall_acc_pre * 100:.2f}%')
    logging.info(f'对抗攻击后 - 总体准确率: {overall_acc_post * 100:.2f}%')

if __name__ == '__main__':
    main()
