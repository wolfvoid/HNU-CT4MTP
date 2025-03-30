import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import os
import joblib  # 用于保存和加载模型
from tqdm import tqdm

np.random.seed(42)


class GBRTModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, checkpoint_path=None, save_model=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.checkpoint_path = checkpoint_path
        self.save_model = save_model

    def train(self, train_loader, feature_columns):
        X_train = []
        y_train = []
        for batch in tqdm(train_loader, desc="Loading training data"):
            x, y = batch
            x = x.numpy()
            y = y.numpy()
            X_train.append(x.reshape(x.shape[0], -1))
            y_train.append(y.reshape(y.shape[0], -1))
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                verbose=1,
            )
        )
        model.fit(X_train, y_train)
        self.models.append(model)
        print("GBRT模型训练完成")

        if self.checkpoint_path and self.save_model == True:
            joblib.dump(self.models, self.checkpoint_path)
            print(f"模型已保存到 {self.checkpoint_path}")

    def predict(self, test_loader, seq_len, pred_len, feature_columns):
        predictions = []
        for batch in tqdm(test_loader, desc="Predicting with GBRT"):
            x, _ = batch
            x = x.numpy()
            # print(f"Processing batch shape: {x.shape}")  # 调试信息
            batch_predictions = np.zeros((x.shape[0], pred_len, x.shape[2]))
            for j in range(x.shape[0]):
                last_seq = x[j, -seq_len:, :].reshape(1, -1)
                forecast = self.models[0].predict(last_seq)
                batch_predictions[j, :, :] = forecast.reshape(pred_len, -1)
            predictions.append(batch_predictions)
        return np.concatenate(predictions, axis=0)

    def evaluate(self, true_values, predictions, loss_save_path=None):
        mse = mean_squared_error(true_values.flatten(), predictions.flatten())
        mae = mean_absolute_error(true_values.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)

        print(f"GBRT模型的MSE: {mse}")
        print(f"GBRT模型的MAE: {mae}")
        print(f"GBRT模型的RMSE: {rmse}")

        if loss_save_path:
            loss_df = pd.DataFrame({
                "MSE": [mse],
                "MAE": [mae],
                "RMSE": [rmse]
            })
            loss_df.to_csv(loss_save_path, index=False)
            print(f"损失值已保存到 {loss_save_path}")

        return mse, mae, rmse


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import CSVDataset

    feature_columns = ["p (mbar)", "T (degC)", "Tpot (K)"]
    seq_len = 16
    pred_len = 16
    os.makedirs("/HNU-CT4MTP/ckpts/gbrt_ckpt", exist_ok=True)
    checkpoint_path = "/HNU-CT4MTP/ckpts/gbrt_ckpt/weather_n=100_16_16.pkl"
    loss_save_path = "/HNU-CT4MTP/ckpts/gbrt_loss.csv"

    train_data_path = "/HNU-CT4MTP/datasets/test_dataset/weather/weather_train.csv"
    test_data_path = "/HNU-CT4MTP/datasets/test_dataset/weather/weather_test.csv"

    train_dataset = CSVDataset(
        csv_path=train_data_path, seq_len=seq_len, valid=False, feature_columns=feature_columns)
    test_dataset = CSVDataset(
        csv_path=test_data_path, seq_len=seq_len, valid=True, feature_columns=feature_columns)
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    gbrt_model = GBRTModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        checkpoint_path=checkpoint_path
    )
    gbrt_model.train(train_loader, feature_columns)
    print("Training completed. Starting prediction.")  # 调试信息
    true_values = []
    for batch in test_loader:
        _, y = batch
        true_values.append(y.numpy())
    true_values = np.concatenate(true_values, axis=0)
    predictions = gbrt_model.predict(
        test_loader, seq_len, pred_len, feature_columns)
    print("Prediction completed.")  # 调试信息
    mse, mae, rmse = gbrt_model.evaluate(
        true_values, predictions, loss_save_path)
    print("\n最终评估结果：")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
