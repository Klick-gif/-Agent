import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import glob
from transformers import GPT2Model
from peft import get_peft_model, LoraConfig, TaskType


# 1. 数据预处理类
class TimeSeriesProcessor:
    def __init__(self, window_size=36, stride=6):
        self.window_size = window_size
        self.stride = stride
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # 目标值缩放
        self.fitted = False

    def fit_transform_train(self, raw_data):
        """处理训练数据并拟合转换器"""
        features = raw_data[:, 2:]  # 特征列
        targets = raw_data[:, 1].astype(float).reshape(-1, 1)  # 目标列（回归问题）

        # 拟合并转换特征
        self.feature_scaler.fit(features)
        X = self.feature_scaler.transform(features)

        # 拟合并转换目标值
        self.target_scaler.fit(targets)
        y = self.target_scaler.transform(targets)

        # 滑动窗口分割
        X_seg, y_seg = [], []
        for i in range(0, len(X) - self.window_size + 1, self.stride):
            X_seg.append(X[i:i + self.window_size])
            y_seg.append(y[i + self.window_size - 1])

        self.fitted = True
        return np.array(X_seg), np.array(y_seg)

    def transform_test(self, raw_data):
        """使用训练时拟合的转换器处理测试数据"""
        if not self.fitted:
            raise ValueError("必须先使用fit_transform_train方法处理训练数据")

        features = raw_data[:, 2:]  # 特征列
        targets = raw_data[:, 1].astype(float).reshape(-1, 1)  # 目标列（回归问题）

        # 转换特征和目标值
        X = self.feature_scaler.transform(features)
        y = self.target_scaler.transform(targets)

        # 滑动窗口分割
        X_seg, y_seg = [], []
        for i in range(0, len(X) - self.window_size + 1, self.stride):
            X_seg.append(X[i:i + self.window_size])
            y_seg.append(y[i + self.window_size - 1])

        return np.array(X_seg), np.array(y_seg)


# 2. 输入嵌入层
class MultivariateTimeSeriesEmbedding(nn.Module):
    def __init__(self, feature_dim=4):
        super().__init__()
        self.hidden_size = 1024  # GPT-2 Medium 的维度

        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size)
        )

        # 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, 2048, self.hidden_size))

    def forward(self, X_feature):
        # 特征嵌入
        feature_emb = self.feature_proj(X_feature)

        # 添加位置编码
        seq_len = feature_emb.shape[1]
        return feature_emb + self.pos_emb[:, :seq_len, :]


# 3. 模型架构
class GPT2ForRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1024  # GPT-2 Medium 的维度

        # 加载 GPT-2 Medium 模型
        self.backbone = GPT2Model.from_pretrained("gpt2-medium")

        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"]  # GPT-2 中的注意力层
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        self.embedding = MultivariateTimeSeriesEmbedding()

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, 1)  # 输出单个值
        )

        # 冻结大部分层
        self.freeze_layers()

    def freeze_layers(self):
        # 定义可训练的参数名称片段
        trainable_param_buf = ["ln", "wpe", "embedding", "regressor", "lora"]

        # 默认情况下，所有参数都不需要梯度
        for n, p in self.named_parameters():
            p.requires_grad = False

        # 只允许特定的层进行训练
        for n, p in self.named_parameters():
            if any(fp in n for fp in trainable_param_buf):
                p.requires_grad = True
                # print(f"{n} is trainable")

        # # 打印冻结状态
        # for n, p in self.named_parameters():
        #     if not p.requires_grad:
        #         print(f"{n} has been frozen")

    def forward(self, X_feature):
        inputs_embeds = self.embedding(X_feature)
        outputs = self.backbone(inputs_embeds=inputs_embeds)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.regressor(pooled)


# 4. 数据集类（回归问题）
class RegressionDataset(Dataset):
    def __init__(self, X_feature, y):
        self.X_feature = torch.FloatTensor(X_feature)
        self.y = torch.FloatTensor(y)  # 回归问题使用FloatTensor

    def __len__(self):
        return len(self.X_feature)

    def __getitem__(self, idx):
        return self.X_feature[idx], self.y[idx]


# 5. 训练函数（回归问题）
def train_regression_model(model, train_loader, test_loader, num_epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()  # 回归问题使用MSE损失
    scaler = torch.cuda.amp.GradScaler()

    history = {
        'train_loss': [],
        'test_loss': [],
        'mae': [],
        'r2': []
    }

    best_model_path = 'best_regression_model.pth'
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(X)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * X.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)

                loss = criterion(outputs, y)
                test_loss += loss.item() * X.size(0)

                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        test_loss = test_loss / len(test_loader.dataset)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

        history['test_loss'].append(test_loss)
        history['mae'].append(mae)
        history['r2'].append(r2)

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1}, 保存最佳模型，测试损失: {test_loss:.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f}")
        print(f"MAE: {mae:.4f} | R²: {r2:.4f}")
        print("-" * 50)

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='MAE')
    plt.plot(history['r2'], label='R² Score')
    plt.title('Regression Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('regression_training_curves.png')
    plt.close()

    return history


# 6. 评估函数（回归问题）
def evaluate_regression(model, test_loader, processor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    # 反归一化
    all_preds = processor.target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    all_labels = processor.target_scaler.inverse_transform(np.array(all_labels).reshape(-1, 1)).flatten()

    # 计算指标
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f"\n最终回归评估结果:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")

    # 绘制预测值与真实值对比图
    plt.figure(figsize=(12, 6))
    plt.plot(all_labels, label='True Values')
    plt.plot(all_preds, label='Predicted Values')
    plt.title('True vs Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('regression_results.png')
    plt.close()

    return mse, mae, r2


# 7. 主流程
if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 数据加载示例（根据你的实际数据路径修改）
    train_data_folder = r"datasets/soh/1"
    test_data_folder = r"datasets/soh/1"

    # 读取训练数据
    train_data = pd.concat(
        [pd.read_csv(file) for file in glob.glob(os.path.join(train_data_folder, "**", "*.csv"), recursive=True)],
        ignore_index=True)

    # 读取测试数据
    test_data = pd.concat(
        [pd.read_csv(file) for file in glob.glob(os.path.join(test_data_folder, "**", "*.csv"), recursive=True)],
        ignore_index=True)

    # 特征选择（根据你的数据调整）
    selected_features = [
        "capacity", "resistance", "CCCT", "CVCT"

    ]

    # 数据预处理
    processor = TimeSeriesProcessor(window_size=36, stride=6)

    # 处理训练数据
    raw_train = train_data[['date', 'OT'] + selected_features].values
    X_train, y_train = processor.fit_transform_train(raw_train)

    # 处理测试数据
    raw_test = test_data[['date', 'OT'] + selected_features].values
    X_test, y_test = processor.transform_test(raw_test)

    # 创建数据集和数据加载器
    train_dataset = RegressionDataset(X_train, y_train)
    test_dataset = RegressionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = GPT2ForRegression()

    # 训练模型
    print("开始训练回归模型...")
    history = train_regression_model(model, train_loader, test_loader, num_epochs=20, lr=1e-4)

    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_regression_model.pth'))
    evaluate_regression(model, test_loader, processor)