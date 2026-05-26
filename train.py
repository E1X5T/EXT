import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 引入进度条
import matplotlib.pyplot as plt  # 引入画图工具
import os
from dataloader import UniversalDualModalDataset
from model import DualModalBaselineNet

# ==========================================
# 🛠️ 配置区
# ==========================================
# SD636
VEIN_TRAIN_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\sd636\FV\train_set"
FINGER_TRAIN_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\sd636\FP\train_set"

VEIN_VAL_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\sd636\FV\test_set"
FINGER_VAL_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\sd636\FP\test_set"

#Nupt840
VEIN_TRAIN_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Nupt840\FV\train_set"
FINGER_TRAIN_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Nupt840\FP\train_set"

VEIN_VAL_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Nupt840\FV\test_set"
FINGER_VAL_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Nupt840\FP\test_set"

#Hkpu150
VEIN_TRAIN_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Hkpu150\FV\train_set"
FINGER_TRAIN_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Hkpu150\FP\train_set"

VEIN_VAL_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Hkpu150\FV\test_set"
FINGER_VAL_DIR = r"C:\Users\lenovo\Desktop\bio EXTNET\data\Hkpu150\FP\test_set"



NUM_CLASSES = 636
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0001
SAVE_MODEL_PATH = "best_strip_mdfa.pth"


# ==========================================

def plot_curves(history):
    """绘制并保存训练/验证的 Loss 和 Accuracy 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)  # 保存为高清图片
    plt.close()
    print("📊 训练曲线已保存为 'training_curves.png'")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 实验启动！正在使用设备: {device}")

    print("正在加载数据集...")
    train_dataset = UniversalDualModalDataset(vein_dir=VEIN_TRAIN_DIR, finger_dir=FINGER_TRAIN_DIR, is_train=True)
    val_dataset = UniversalDualModalDataset(vein_dir=VEIN_VAL_DIR, finger_dir=FINGER_VAL_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ 数据加载完毕！训练集数量: {len(train_dataset)} | 验证集数量: {len(val_dataset)}")

    model = DualModalBaselineNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 每 30 个 Epoch 学习率乘以 0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_val_acc = 0.0

    # 用于记录画图的数据
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # 使用 tqdm 包装 train_loader
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=False)

        for v_imgs, f_imgs, labels in train_pbar:
            v_imgs, f_imgs, labels = v_imgs.to(device), f_imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(v_imgs, f_imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # 实时更新进度条上的后缀信息
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # 使用 tqdm 包装 val_loader
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", leave=False)

        with torch.no_grad():
            for v_imgs, f_imgs, labels in val_pbar:
                v_imgs, f_imgs, labels = v_imgs.to(device), f_imgs.to(device), labels.to(device)
                outputs = model(v_imgs, f_imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val

        # 记录数据用于画图
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # 1. 打印当前 Epoch 总结
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")

        # 2. 判断并保存最优模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"   🌟 验证集精度提升至 {epoch_val_acc:.2f}%, 模型已保存！")

        # 3. 🔥 在整个 Epoch 的最后安全更新学习率 (仅此一次)
        scheduler.step()

    # 训练结束后，自动画图
    plot_curves(history)
    print("🎉 所有训练已完成！")


if __name__ == '__main__':
    main()