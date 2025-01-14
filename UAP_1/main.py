import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
import logging
from torch.optim.lr_scheduler import StepLR
import numpy as np

# =============================
# 1. 日志配置
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_loading.log"),
        logging.StreamHandler()
    ]
)

# =============================
# 2. 数据集定义
# =============================
class RealFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        self.transform = transform

        self.samples = []
        self.real_count = 0
        self.fake_count = 0

        # 加载 real
        if os.path.isdir(self.real_dir):
            real_classes = sorted(os.listdir(self.real_dir))
            for cls in real_classes:
                cls_path = os.path.join(self.real_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                class_files = os.listdir(cls_path)
                for img_name in class_files:
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, 0))  # 0=real
                        self.real_count += 1
        else:
            logging.error(f"Directory {self.real_dir} does not exist.")

        # 加载 fake
        if os.path.isdir(self.fake_dir):
            fake_classes = sorted(os.listdir(self.fake_dir))
            for cls in fake_classes:
                cls_path = os.path.join(self.fake_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                class_files = os.listdir(cls_path)
                for img_name in class_files:
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, 1))  # 1=fake
                        self.fake_count += 1
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
        except:
            logging.error(f"Error loading image {img_path}. Returning a blank image.")
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================
# 3. 训练函数
# =============================
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

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

# =============================
# 4. 生成 Universal Adversarial Perturbation
# =============================
def generate_universal_perturbation(model, dataloader, device='cpu',
                                    eps=0.003, alpha=0.0008, max_iter=1):
    """
    一个简化的UAP生成示例 (L_infinity约束)：
      - 初始化通用扰动 uap = 0
      - 对datloader中每个batch进行若干轮:
          1) 前向传播得到损失
          2) 计算梯度符号
          3) 更新 uap
      - 将uap裁剪到 [-eps, eps] 范围内
    参数:
      eps: UAP的L_infinity最大范数半径
      alpha: 每次更新的步长
      max_iter: 在所有数据上迭代的轮数(可加大以提升UAP效果)
    返回:
      uap: 形状与图像相同 (NCHW) 中的 (1, C, H, W)
    """
    model.eval()
    # 这里假设输入图像尺寸是 (3, 224, 224)
    # 如果你有更通用需求，需要根据实际尺寸自动获取
    uap = torch.zeros((1, 3, 224, 224), device=device)

    criterion = nn.CrossEntropyLoss()

    for _ in range(max_iter):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 对当前 batch 应用已有的uap
            adv_images = images + uap
            adv_images = torch.clamp(adv_images, 0.0, 1.0)

            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            # 计算梯度并更新uap
            grad_sign = adv_images.grad.data.sign()
            # 只更新uap本身 (images只是中间变量)
            # 这里可以只更新uap的一部分, 但简单起见直接所有像素点都更新
            uap = uap + alpha * grad_sign.mean(dim=0, keepdim=True)

            # 将uap限制在 [-eps, eps]
            uap = torch.clamp(uap, min=-eps, max=eps)

    return uap.detach()

# =============================
# 5. UAP攻击时对单张图像加扰动
# =============================
def uap_attack(images, uap):
    """
    对一个batch的images加上 universal perturbation
    参数:
      images: shape=[B, C, H, W]
      uap:    shape=[1, C, H, W]
    """
    adv_images = images + uap
    adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images

# =============================
# 6. 测试函数
# =============================
def evaluate(model, data_loader, device='cpu', uap=None):
    """
    如果uap不为None, 则对每个batch加上该uap再推断
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if uap is not None:
                images = uap_attack(images, uap)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

def evaluate_real_fake(model, test_real_loader, test_fake_loader, device='cpu', uap=None):
    # 对 real dataset 测
    real_acc = evaluate(model, test_real_loader, device=device, uap=uap)
    # 对 fake dataset 测
    fake_acc = evaluate(model, test_fake_loader, device=device, uap=uap)
    return real_acc, fake_acc

# =============================
# 7. 主流程
# =============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'使用设备: {device}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    dataset = RealFakeDataset(root_dir='../data', transform=transform)
    logging.info(f'数据集总大小: {len(dataset)}')

    # 简单检查
    expected_total = 4000 * 2
    if len(dataset) != expected_total:
        logging.warning(f"数据集大小为 {len(dataset)}，与预期 {expected_total} 不符，请检查数据集。")

    # 划分 train / test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 从 test_dataset 中分离 real / fake
    test_real_indices = [i for i, (img, lbl) in enumerate(test_dataset) if lbl == 0]
    test_fake_indices = [i for i, (img, lbl) in enumerate(test_dataset) if lbl == 1]

    test_real_dataset = Subset(test_dataset, test_real_indices)
    test_fake_dataset = Subset(test_dataset, test_fake_indices)
    logging.info(f'测试集: real={len(test_real_dataset)}, fake={len(test_fake_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_real_loader = DataLoader(test_real_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_fake_loader = DataLoader(test_fake_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 使用ResNet18
    model = models.resnet18(pretrained=True)
    # 改成二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)

    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 是否已有模型文件
    model_path = 'model.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info('成功加载已有模型')
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            logging.info('重新训练模型...')
            train_model(model, train_loader, criterion, optimizer, scheduler,
                        num_epochs=20, device=device)
            torch.save(model.state_dict(), model_path)
            logging.info(f'模型已保存 {model_path}')
    else:
        # 训练
        train_model(model, train_loader, criterion, optimizer, scheduler,
                    num_epochs=20, device=device)
        torch.save(model.state_dict(), model_path)
        logging.info(f'模型已保存 {model_path}')

    # ============= 评估 无UAP(攻击前) =============
    real_acc_pre, fake_acc_pre = evaluate_real_fake(
        model, test_real_loader, test_fake_loader, device=device, uap=None
    )
    logging.info(f'无UAP - 真实图片准确率: {real_acc_pre*100:.2f}%, 假图片准确率: {fake_acc_pre*100:.2f}%')

    # ============= 生成 UAP =============
    # 注意：可只用 train_loader 或 test_loader 来生成 UAP
    #       这里简单用 train_loader 全部数据
    #       可调 eps/alpha/max_iter 获取更强/更弱的扰动
    uap = generate_universal_perturbation(
        model, train_loader, device=device,
        eps=0.003,       # L_infinity 范数
        alpha=0.0008,    # 步长
        max_iter=1       # 在所有数据上迭代1轮，可适当加大
    )
    logging.info("UAP已生成。")

    # ============= 评估 加UAP(攻击后) =============
    real_acc_post, fake_acc_post = evaluate_real_fake(
        model, test_real_loader, test_fake_loader, device=device, uap=uap
    )
    logging.info(f'UAP攻击后 - 真实图片准确率: {real_acc_post*100:.2f}%, 假图片准确率: {fake_acc_post*100:.2f}%')

    # 可选: 计算总体准确率
    total_real = len(test_real_dataset)
    total_fake = len(test_fake_dataset)
    total_num = total_real + total_fake
    overall_pre = (real_acc_pre * total_real + fake_acc_pre * total_fake) / total_num
    overall_post = (real_acc_post * total_real + fake_acc_post * total_fake) / total_num
    logging.info(f'无UAP - 总体准确率: {overall_pre*100:.2f}%')
    logging.info(f'UAP后 - 总体准确率: {overall_post*100:.2f}%')

if __name__ == '__main__':
    main()
