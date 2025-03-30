import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pandas as pd
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from model.CasualTransformer import CausalTransformer
from einops import rearrange
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model.dataset import CSVDataset

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)


class TransformerLightningModule(pl.LightningModule):
    """PyTorch Lightning 模块，封装 CausalTransformer 并添加高斯混合模型支持"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_features,
        num_layers,
        num_heads,
        warmup_steps,
        stable_steps,
        decay_steps,
        num_gmm_kernels=5,
        learning_rate=0.001,
        ff_dim=None,
        dropout=0.1,
        gamma=0.9999,
        feature_means=None,
        feature_stds=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_features = num_features
        self.num_gmm_kernels = num_gmm_kernels

        # 保存特征的统计信息
        self.register_buffer("feature_means", torch.FloatTensor(feature_means))
        self.register_buffer("feature_stds", torch.FloatTensor(feature_stds))

        # 创建模型
        self.model = CausalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features
            * num_gmm_kernels
            * 2,  # 每个特征的每个高斯核都有均值和方差
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def normalize(self, x):
        """对输入数据进行归一化"""
        return (x - self.feature_means) / (self.feature_stds + 1e-8)

    def denormalize(self, x):
        """还原归一化的数据"""
        return x * self.feature_stds + self.feature_means

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 归一化输入数据
        x_normalized = self.normalize(x)

        # 获取transformer的输出
        output = self.model(
            x_normalized
        )  # (batch_size, seq_len, num_features * num_gmm_kernels * 2)

        batch_size, seq_len, _ = output.shape

        # 重塑输出以分离均值和方差
        output = output.reshape(
            batch_size, seq_len, self.num_features, self.num_gmm_kernels, 2
        )
        # (batch_size, seq_len, num_features, num_gmm_kernels)
        mu = output[..., 0]
        # (batch_size, seq_len, num_features, num_gmm_kernels)
        logvar = output[..., 1]
        # 还原mu（均值）- 直接使用denormalize
        mu_denorm = self.denormalize(
            rearrange(mu, "b s f k -> (k b) s f", k=self.num_gmm_kernels)
        )  # (batch_size, seq_len, num_features, num_gmm_kernels)
        mu_denorm = rearrange(
            mu_denorm, "(k b) s f -> b s f k", k=self.num_gmm_kernels)

        # 还原logvar（对数方差）- 需要考虑标准差的平方
        # 由于logvar是对数方差，我们需要调整还原方式
        logvar_denorm = logvar + 2 * torch.log(
            # self.feature_stds.unsqueeze(-1)
            torch.clamp(self.feature_stds.unsqueeze(-1), min=1e-6)
        )  # 广播到num_gmm_kernels维度

        # 使用还原后的参数进行重参数化采样
        res = self.reparameterize(
            mu_denorm, logvar_denorm
        )  # (batch_size, seq_len, num_features, num_gmm_kernels)
        res = res.mean(dim=-1)  # (batch_size, seq_len, num_features)
        # 防止标准差为零或过小的情况导致数值不稳定或计算错误。
        # 确保对数计算中不会出现对数零或负数的情况，从而提高代码的鲁棒性和稳定性

        return res, mu_denorm, logvar_denorm

    def gaussian_nll_loss(self, y_true, mu, logvar):
        """计算高斯负对数似然损失"""
        var = torch.exp(logvar)
        # 添加下限方止除以0
        var = torch.clamp(var, min=1e-6)
        nll = 0.5 * (
            torch.log(2 * np.pi * var) + (y_true.unsqueeze(-1) - mu) ** 2 / var
        )
        return nll.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)

        # 计算重构损失（MSE）
        recon_loss = F.mse_loss(y_hat, y)

        # 计算每个高斯核的负对数似然损失
        nll_loss = self.gaussian_nll_loss(y, mu, logvar)

        # 总损失
        loss = recon_loss + nll_loss

        self.log("train_loss", loss, prog_bar=True,
                 sync_dist=True, on_epoch=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_nll_loss", nll_loss)
        self.log("learning_rate",
                 self.trainer.optimizers[0].param_groups[0]["lr"])
        return recon_loss / (1e-6 + recon_loss.detach()) + nll_loss / (
            1e-6 + nll_loss.detach()
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)

        recon_loss = F.mse_loss(y_hat, y)
        nll_loss = self.gaussian_nll_loss(y, mu, logvar)
        loss = recon_loss + nll_loss

        self.log("val_loss", loss, prog_bar=True,
                 sync_dist=True, on_epoch=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_nll_loss", nll_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)

        # recon_loss = F.mse_loss(y_hat, y)
        # nll_loss = self.gaussian_nll_loss(y, mu, logvar)
        # loss = recon_loss + nll_loss

        # self.log("test_loss", loss)
        # self.log("test_recon_loss", recon_loss)
        # self.log("test_nll_loss", nll_loss)
        # return loss
        # 计算重构损失（MSE）
        mse_loss = F.mse_loss(y_hat, y)
        rmse_loss = torch.sqrt(mse_loss)  # 均方根误差
        mae_loss = F.l1_loss(y_hat, y)  # 平均绝对误差

        # 计算每个高斯核的负对数似然损失
        nll_loss = self.gaussian_nll_loss(y, mu, logvar)
        loss = mse_loss + nll_loss

        # 记录测试损失
        self.log("test_loss", loss)
        self.log("test_mse_loss", mse_loss)
        self.log("test_rmse_loss", rmse_loss)
        self.log("test_mae_loss", mae_loss)
        self.log("test_nll_loss", nll_loss)

        return {"test_loss": loss, "test_mse_loss": mse_loss, "test_rmse_loss": rmse_loss, "test_mae_loss": mae_loss}

    def configure_optimizers(self):
        weight_decay = 0.05
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=weight_decay)

        def warmup_stable_decay_lr(step):
            if step < self.hparams.warmup_steps:
                # Warmup阶段：线性增加学习率
                return float(step) / float(max(1, self.hparams.warmup_steps))
            elif step < self.hparams.warmup_steps + self.hparams.stable_steps:
                # Stable阶段：保持学习率不变
                return 1.0
            else:
                # Decay阶段：快速衰减学习率（例如使用余弦衰减）
                # return self.hparams.gamma ** ((step - self.hparams.warmup_steps))
                now_decay_steps = step - self.hparams.warmup_steps - self.hparams.stable_steps
                progress = now_decay_steps / self.hparams.decay_steps
                progress_tensor = torch.tensor(progress, dtype=torch.float32)
                return 0.5 * (1 + torch.cos(torch.pi * progress_tensor))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_stable_decay_lr)

        # # 创建学习率调度器
        # def warmup_exponential_lr(step):
        #     if step < self.hparams.warmup_steps:
        #         # 线性预热
        #         return float(step) / float(max(1, self.hparams.warmup_steps))
        #     else:
        #         # 指数衰减
        #         return self.hparams.gamma ** ((step - self.hparams.warmup_steps))

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=warmup_exponential_lr
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def predict_sequence(self, initial_seq, pred_len):
        """使用模型进行自回归预测"""
        self.eval()  # 确保模型处于评估模式

        with torch.no_grad():
            # 初始序列
            curr_seq = torch.FloatTensor(initial_seq).unsqueeze(
                0
            )  # (1, init_len, num_features)
            output_seq = initial_seq.copy()  # 使用copy避免修改原始数据

            # 当前输入序列
            input_seq = curr_seq.clone()    # 使用.clone()来复制张量

            # 逐步预测
            for i in range(pred_len):
                # 使用当前序列预测下一个值
                pred, _, _ = self(input_seq)
                next_val = pred[:, -1, :].cpu().numpy()

                # 将预测值添加到输出序列
                output_seq = np.vstack([output_seq, next_val[0]])

                # 更新输入序列（滑动窗口）
                input_seq = torch.FloatTensor(
                    output_seq[-len(initial_seq):]
                ).unsqueeze(0)

        return output_seq


class CasualTransformer_train_test():
    def __init__(self, train_csv, test_csv, feature_columns, seq_len, pred_len, max_epochs, ckpt_path):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.max_epochs = max_epochs

        self.warmup_ratio = 0.05
        self.stable_ratio = 0.6
        self.val_check_ratio = 0.02

        self.ckpt_path = ckpt_path
        os.makedirs(self.ckpt_path, exist_ok=True)
        if not os.path.exists(ckpt_path):
            raise ("ckpt path not exist")

    def load_test_sequence(self, csv_file, seq_len, pred_len, feature_columns):
        """从CSV文件加载测试序列"""
        df = pd.read_csv(csv_file)

        # 检查所有指定的特征列是否存在
        missing_cols = [
            col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"在文件 {csv_file} 中未找到以下特征列: {missing_cols}")

        # 提取特征数据
        features = df[feature_columns].values  # (seq_len, num_features)

        # 获取初始序列和完整序列
        initial_seq = features[:seq_len]  # (seq_len, num_features)
        full_seq = features  # (total_len, num_features)

        return initial_seq, full_seq

    def plot_results(self,
                     true_seq,
                     pred_seq,
                     feature_columns,
                     feature_idx=0,
                     pred_len=20,
                     title="Prediction Results",
                     ):
        """绘制真实序列和预测序列的对比图"""
        plt.figure(figsize=(12, 6))

        # 选择要绘制的特征
        true_feature = true_seq[:, feature_idx]
        pred_feature = pred_seq[:, feature_idx]

        # 绘制真实值
        plt.plot(
            np.arange(len(true_feature)), true_feature, label="Ground Truth", color="blue"
        )

        # 绘制预测值（包括初始序列和预测部分）
        # 初始序列长度就是seq_len（在main函数中设置为100）
        initial_len = len(pred_feature) - pred_len  # 使用pred_len作为预测长度
        plt.plot(
            np.arange(len(pred_feature)),
            pred_feature,
            label="Prediction",
            color="red",
            linestyle="--",
        )

        # 标记分隔线
        plt.axvline(
            x=initial_len - 1, color="green", linestyle="-", label="Prediction Start"
        )

        plt.title(f"{title} - {feature_columns[feature_idx]}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"prediction_result_{feature_columns[feature_idx]}.png")
        # plt.show()

    def main(self,):
        # 参数设置
        seq_len = self.seq_len  # 输入序列长度
        batch_size = 256
        max_epochs = self.max_epochs

        # 设置特征列
        feature_columns = self.feature_columns

        # 检查数据目录是否存在
        # if not os.path.exists("data"):
        #     print("Data directory does not exist, generating data...")
        #     import generate_csv_data
        #     generate_csv_data.main()

        # 创建数据集
        print("Loading datasets...")
        train_dataset = CSVDataset(
            csv_path=self.train_csv,
            seq_len=seq_len,
            valid=False,
            feature_columns=feature_columns,
        )
        val_dataset = CSVDataset(
            csv_path=self.train_csv,
            seq_len=seq_len,
            valid=True,
            feature_columns=feature_columns,
        )
        test_dataset = CSVDataset(
            csv_path=self.test_csv,
            seq_len=seq_len,
            valid=True,
            feature_columns=feature_columns,
        )

        # 获取特征维度
        sample_x, _ = train_dataset[0]
        input_dim = sample_x.shape[1]  # 特征数量
        output_dim = input_dim  # 输出维度与输入维度相同

        # 模型参数
        hidden_dim = 512
        num_layers = 10
        num_heads = 8  # 每个head的最佳hidden dim是8.33*ln(seq_len)~=52

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=63, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=63, pin_memory=True)
        test_loader = DataLoader(
            test_dataset, batch_size=1, num_workers=63, pin_memory=True)

        # 计算各部分调度步数
        print("computing lr stage steps...")
        total_steps = len(train_loader) * self.max_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        stable_steps = int(total_steps * self.stable_ratio)
        decay_steps = total_steps - warmup_steps - stable_steps
        val_check_interval = min(
            int(total_steps * self.val_check_ratio), len(train_loader))
        print(f"total_steps={total_steps}")
        print(f"warmup_steps={warmup_steps}")
        print(f"stable_steps={stable_steps}")
        print(f"decay_steps={decay_steps}")
        print(f"val_check_interval={val_check_interval}")
        # warmup_steps = 2000
        # val_check_interval = 20
        # stable_steps = 0
        # decay_steps = 0

        # 创建模型
        print("Initializing model...")
        model = TransformerLightningModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_features=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            learning_rate=8e-4,
            dropout=0.1,
            warmup_steps=warmup_steps,
            gamma=0.9999,
            feature_means=train_dataset.feature_means,  # 传递训练集的统计信息
            feature_stds=train_dataset.feature_stds,
        )

        # 设置回调
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.ckpt_path,
            filename="transformer-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        )

        # 设置日志记录器
        logger = TensorBoardLogger("lightning_logs", name="transformer")

        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
            log_every_n_steps=10,
            accelerator="auto",
            # precision="16-mixed",
            precision="32",
            val_check_interval=val_check_interval,
            gradient_clip_val=1.0,  # 添加梯度裁剪阈值
            gradient_clip_algorithm="norm",  # 使用L2范数裁剪
        )

        # 训练模型
        print("Starting model training...")
        trainer.fit(model, train_loader, val_loader)

        # 测试模型
        pred_len = seq_len  # 预测序列长度
        print("Testing model...")
        test_results = trainer.test(model, test_loader)

        # 打印测试结果
        print("Test Results:")
        for result in test_results:
            print(f"Test Loss: {result['test_loss']:.4f}")
            print(f"MSE: {result['test_mse_loss']:.4f}")
            print(f"MAE: {result['test_mae_loss']:.4f}")
            print(f"RMSE: {result['test_rmse_loss']:.4f}")

        # 生成预测
        # print("Generating predictions...")
        # 从测试集加载一个样本
        # initial_seq, true_seq = self.load_test_sequence(
        #     self.test_csv, seq_len, pred_len, feature_columns
        # )

        # 使用模型预测
        # pred_seq = model.predict_sequence(initial_seq, pred_len)

        # 为每个特征绘制结果
        # num_features = len(feature_columns)
        # for i in range(num_features):
        #     self.plot_results(
        #         true_seq,
        #         pred_seq,
        #         feature_columns,
        #         feature_idx=i,
        #         pred_len=pred_len,
        #         title="Transformer Time Series Prediction",
        #     )

        # print(f"完成！预测结果已保存为对应特征名称的PNG文件")


if __name__ == "__main__":
    run = CasualTransformer_train_test(
        train_csv="/HNU-CT4MTP/datasets/test_dataset/exchange_rate/exchange_rate_train.csv",
        test_csv="/HNU-CT4MTP/datasets/test_dataset/exchange_rate/exchange_rate_test.csv",
        feature_columns=["0", "1", "2"],
        seq_len=16,
        pred_len=None,
        max_epochs=15,
        ckpt_path="",
    )
    run.main()
