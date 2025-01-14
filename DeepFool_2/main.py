import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
import logging
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import torchattacks

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
    """
    在 data/ 文件夹下:
        real/ 下若干子文件夹(各类), fake/ 下若干子文件夹(各类),
        这里仅区分 real(标签0) vs fake(标签1)。
    """
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
            logging.info(f"Found {len(real_classes)} classes in real directory.")
            for cls in real_classes:
                cls_path = os.path.join(self.real_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                class_files = os.listdir(cls_path)
                logging.info(f"Loading images from class '{cls}' in real. Total images: {len(class_files)}")
                for img_name in class_files:
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, 0))  # label=0
                        self.real_count += 1
        else:
            logging.error(f"Directory {self.real_dir} does not exist.")

        # 加载 fake
        if os.path.isdir(self.fake_dir):
            fake_classes = sorted(os.listdir(self.fake_dir))
            logging.info(f"Found {len(fake_classes)} classes in fake directory.")
            for cls in fake_classes:
                cls_path = os.path.join(self.fake_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                class_files = os.listdir(cls_path)
                logging.info(f"Loading images from class '{cls}' in fake. Total images: {len(class_files)}")
                for img_name in class_files:
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, 1))  # label=1
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
            logging.error(f"Error loading image {img_path}, returning a blank image.")
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
            images = images.to(device)
            labels = labels.to(device)

            # 前向
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向 & 优化
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
        epoch_acc = (correct / total) * 100
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    logging.info('训练完成')

# =============================
# 4. 推断 & 对抗攻击评估
# =============================
def evaluate(model, data_loader, attack=None, device='cpu'):
    """
    如果 attack 不为 None，则使用对抗攻击对当前 batch 图像进行扰动，再推断
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.enable_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            if attack:
                # 攻击时，这里会对images做DeepFool
                images = attack(images, labels)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc

def evaluate_real_fake(model, test_real_loader, test_fake_loader, attack_real=None, attack_fake=None, device='cpu'):
    # 评估真实图片
    if attack_real is not None:
        real_acc = evaluate(model, test_real_loader, attack=attack_real, device=device)
    else:
        real_acc = evaluate(model, test_real_loader, attack=None, device=device)

    # 评估假图片
    if attack_fake is not None:
        fake_acc = evaluate(model, test_fake_loader, attack=attack_fake, device=device)
    else:
        fake_acc = evaluate(model, test_fake_loader, attack=None, device=device)

    return real_acc, fake_acc

# =============================
# 5. 主流程
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
    expected_total = 8000  # real4000 + fake4000
    if len(dataset) != expected_total:
        logging.warning(f"数据集大小 {len(dataset)} != 预期 {expected_total}, 请检查数据.")

    # 划分 train / test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 从 test_dataset 中找出 real / fake
    test_real_indices = [i for i, (img, lbl) in enumerate(test_dataset) if lbl == 0]
    test_fake_indices = [i for i, (img, lbl) in enumerate(test_dataset) if lbl == 1]

    test_real_dataset = Subset(test_dataset, test_real_indices)
    test_fake_dataset = Subset(test_dataset, test_fake_indices)

    logging.info(f'测试集: real={len(test_real_dataset)}, fake={len(test_fake_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_real_loader = DataLoader(test_real_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_fake_loader = DataLoader(test_fake_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 定义模型
    model = models.resnet18(pretrained=True)
    # 改成二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)

    # 定义loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 先尝试加载已训练的模型
    model_path = 'model.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info('成功加载已有模型')
        except Exception as e:
            logging.error(f"加载模型失败: {e}, 将重新训练")
            train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)
            torch.save(model.state_dict(), model_path)
            logging.info('模型已保存为 model.pth')
    else:
        # 若没有则训练并保存
        logging.info('开始训练模型...')
        train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)
        torch.save(model.state_dict(), model_path)
        logging.info('模型已保存为 model.pth')

    # ========== 1) 无攻击下准确率 =============
    real_acc_pre, fake_acc_pre = evaluate_real_fake(
        model,
        test_real_loader,
        test_fake_loader,
        attack_real=None,
        attack_fake=None,
        device=device
    )
    logging.info(f'无攻击 - 真实图片准确率: {real_acc_pre*100:.2f}%, 假图片准确率: {fake_acc_pre*100:.2f}%')

    # ========== 2) 使用 DeepFool 进行攻击 =============
    # 参数可调, 比如 steps=50, overshoot=0.02
    # steps 越大, 攻击越强; overshoot 决定了越过决策边界的量
    deepfool_real = torchattacks.DeepFool(model, steps=5, overshoot=0.02)
    deepfool_fake = torchattacks.DeepFool(model, steps=5, overshoot=0.02)

    real_acc_post, fake_acc_post = evaluate_real_fake(
        model,
        test_real_loader,
        test_fake_loader,
        attack_real=deepfool_real,
        attack_fake=deepfool_fake,
        device=device
    )
    logging.info(f'DeepFool攻击 - 真实图片准确率: {real_acc_post*100:.2f}%, 假图片准确率: {fake_acc_post*100:.2f}%')

    # 总体准确率(可选)
    total_real = len(test_real_dataset)
    total_fake = len(test_fake_dataset)
    total_num = total_real + total_fake
    overall_pre = (real_acc_pre * total_real + fake_acc_pre * total_fake) / total_num
    overall_post = (real_acc_post * total_real + fake_acc_post * total_fake) / total_num
    logging.info(f'无攻击 - 总体准确率: {overall_pre*100:.2f}%')
    logging.info(f'DeepFool后 - 总体准确率: {overall_post*100:.2f}%')

if __name__ == '__main__':
    main()
