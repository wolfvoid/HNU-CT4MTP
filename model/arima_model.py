import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import os
import joblib  # 用于保存和加载模型
from tqdm import tqdm

np.random.seed(42)


class ARIMAModel:
    def __init__(self, order=(5, 1, 0), checkpoint_path=None, save_model=False):
        """
        初始化ARIMA模型
        :param order: ARIMA模型的参数 (p, d, q)
        :param checkpoint_path: 模型检查点保存路径
        """
        self.order = order
        self.models = []
        self.checkpoint_path = checkpoint_path
        self.save_model = save_model

    def train(self, train_loader, feature_columns):
        """
        训练ARIMA模型
        :param train_loader: DataLoader，用于加载训练数据
        :param feature_columns: 特征列名列表
        """
        for batch in tqdm(train_loader, desc="Training ARIMA"):
            x, _ = batch
            x = x.numpy()  # 将Tensor转换为numpy数组
            for i in range(x.shape[2]):  # 遍历每个特征
                feature_data = x[:, :, i].flatten()  # 将所有时间步的数据展平
                model = ARIMA(feature_data, order=self.order)
                model_fit = model.fit()
                self.models.append(model_fit)
                # print(f"特征 {feature_columns[i]} 的ARIMA模型训练完成")

        # 保存模型检查点
        if self.checkpoint_path and self.save_model == True:
            joblib.dump(self.models, self.checkpoint_path)
            print(f"模型已保存到 {self.checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        从检查点加载模型
        :param checkpoint_path: 模型检查点路径
        """
        self.models = joblib.load(checkpoint_path)
        print(f"模型已从 {checkpoint_path} 加载")

    def predict(self, test_loader, seq_len, pred_len, feature_columns):
        """
        使用训练好的ARIMA模型进行预测
        :param test_loader: DataLoader，用于加载测试数据
        :param seq_len: 输入序列长度
        :param pred_len: 预测序列长度
        :param feature_columns: 特征列名列表
        :return: 预测结果
        """
        predictions = []
        for batch in tqdm(test_loader, desc="Predicting with ARIMA"):
            x, _ = batch
            x = x.numpy()  # 将Tensor转换为numpy数组
            batch_predictions = np.zeros((x.shape[0], pred_len, x.shape[2]))
            for i in range(x.shape[2]):  # 遍历每个特征
                model_fit = self.models[i]
                for j in range(x.shape[0]):  # 遍历每个样本
                    last_seq = x[j, -seq_len:, i].flatten()  # 获取最后一个序列
                    model_fit = model_fit.apply(last_seq)  # 更新模型状态
                    forecast = model_fit.forecast(steps=pred_len)
                    batch_predictions[j, :, i] = forecast
            predictions.append(batch_predictions)
        return np.concatenate(predictions, axis=0)

    def evaluate(self, true_values, predictions, loss_save_path=None):
        """
        评估模型性能
        :param true_values: 真实值
        :param predictions: 预测值
        :param loss_save_path: 损失值保存路径
        :return: MSE, MAE, RMSE
        """
        mse = mean_squared_error(true_values.flatten(), predictions.flatten())
        mae = mean_absolute_error(true_values.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)

        # 打印评估结果
        print(f"ARIMA模型的MSE: {mse}")
        print(f"ARIMA模型的MAE: {mae}")
        print(f"ARIMA模型的RMSE: {rmse}")

        # 保存损失值
        if loss_save_path:
            loss_df = pd.DataFrame({
                "MSE": [mse],
                "MAE": [mae],
                "RMSE": [rmse]
            })
            loss_df.to_csv(loss_save_path, index=False)
            print(f"损失值已保存到 {loss_save_path}")

        return mse, mae, rmse


# 示例用法
if __name__ == "__main__":
    # 假设已经有一个DataLoader
    from torch.utils.data import DataLoader
    from dataset import CSVDataset

    feature_columns = ["p (mbar)", "T (degC)", "Tpot (K)"]
    seq_len = 16
    pred_len = 16
    checkpoint_path = "/HNU-CT4MTP/ckpts/arima_ckpt/weather_5_1_1_16_16.pkl"
    loss_save_path = "/HNU-CT4MTP/ckpts/arima_loss.csv"  # 损失值保存路径

    # 创建数据集和DataLoader
    train_data_path = "/HNU-CT4MTP/datasets/test_dataset/weather/weather_train.csv"
    test_data_path = "/HNU-CT4MTP/datasets/test_dataset/weather/weather_test.csv"
    train_dataset = CSVDataset(
        csv_path=train_data_path, seq_len=seq_len, valid=False, feature_columns=feature_columns)
    test_dataset = CSVDataset(
        csv_path=test_data_path, seq_len=seq_len, valid=True, feature_columns=feature_columns)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # 初始化并训练ARIMA模型
    arima_model = ARIMAModel(order=(5, 1, 1), checkpoint_path=checkpoint_path)
    arima_model.train(train_loader, feature_columns)

    # 从检查点加载模型
    arima_model.load_checkpoint(checkpoint_path)

    # 进行预测
    true_values = []
    for batch in test_loader:
        _, y = batch
        true_values.append(y.numpy())
    true_values = np.concatenate(true_values, axis=0)
    predictions = arima_model.predict(
        test_loader, seq_len, pred_len, feature_columns)

    # 评估模型
    mse, mae, rmse = arima_model.evaluate(
        true_values, predictions, loss_save_path)

    # 最终输出
    print("\n最终评估结果：")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
