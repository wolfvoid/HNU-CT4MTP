from model.dataset import CSVDataset
from runCasualTransformer import CasualTransformer_train_test
from model.arima_model import ARIMAModel
from model.gbrt_model import GBRTModel
import numpy as np
from torch.utils.data import DataLoader
import os
import joblib  # 用于保存和加载模型
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append("/HNU-CT4MTP")

np.random.seed(42)

# models


def run(model_name, dataset_name, seq_len, pred_len):
    root_dir = f"/HNU-CT4MTP/ckpts/{model_name}_ckpt"
    os.makedirs(root_dir, exist_ok=True)
    train_data_path = f"/HNU-CT4MTP/5MTP-datasets/{dataset_name}/{dataset_name}_train.csv"
    test_data_path = f"/HNU-CT4MTP/5MTP-datasets/{dataset_name}/{dataset_name}_test.csv"
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"文件 {train_data_path} 不存在！")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"文件 {test_data_path} 不存在！")

    if dataset_name == "electricity":
        # feature(321): 0-319, OT   train_line:23351    test_line:2955
        feature_columns = ["0", "1", "2"]
    elif dataset_name == "exchange_rate":
        # feature(8): 0-6, OT       train_line:6453     test_line:1137
        feature_columns = ["0", "1", "2"]
    elif dataset_name == "PSM":
        # feature(25): feature_0 - feature_24    train_line:132482     test_line:87842
        feature_columns = ["feature_0", "feature_1", "feature_2"]
    elif dataset_name == "traffic":
        # feature(862):0 - 860, OT  train_line:15335    test_line:2211
        feature_columns = ["0", "1", "2"]
    elif dataset_name == "weather":
        # feature(21):p (mbar),T (degC),Tpot (K),Tdew (degC),rh (%),VPmax (mbar),VPact (mbar),VPdef (mbar),sh (g/kg),H2OC (mmol/mol),rho (g/m**3),wv (m/s),max. wv (m/s),wd (deg),rain (mm),raining (s),SWDR (W/m�),PAR (�mol/m�/s),max. PAR (�mol/m�/s),Tlog (degC),OT  train_line:48232    test_line:4466
        feature_columns = ["p (mbar)", "T (degC)", "Tpot (K)"]
    else:
        raise ("dataset wrong")

    if model_name != "casual":
        train_dataset = CSVDataset(
            csv_path=train_data_path, seq_len=seq_len, valid=False, feature_columns=feature_columns)
        test_dataset = CSVDataset(
            csv_path=test_data_path, seq_len=seq_len, valid=True, feature_columns=feature_columns)
        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    if model_name == "gbrt":
        n_num = 3
        checkpoint_path = os.path.join(
            root_dir, f"{dataset_name}_n={n_num}_{seq_len}_{pred_len}.pkl")
        loss_save_path = os.path.join(
            root_dir, f"{dataset_name}_n={n_num}_{seq_len}_{pred_len}.csv")
        gbrt_model = GBRTModel(
            n_estimators=n_num,
            learning_rate=0.1,
            max_depth=3,
            checkpoint_path=checkpoint_path,
            save_model=False,
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
    elif model_name == "arima":
        p, d, q = 3, 1, 1
        checkpoint_path = os.path.join(
            root_dir, f"{dataset_name}_{p}_{d}_{q}_{seq_len}_{pred_len}.pkl")
        loss_save_path = os.path.join(
            root_dir, f"{dataset_name}_{p}_{d}_{q}_{seq_len}_{pred_len}.csv")
        # 初始化并训练ARIMA模型
        arima_model = ARIMAModel(
            order=(p, d, q), checkpoint_path=checkpoint_path, save_model=False)
        arima_model.train(train_loader, feature_columns)
        # 从检查点加载模型
        # arima_model.load_checkpoint(checkpoint_path)
        # 进行预测
        true_values = []
        for batch in test_loader:
            _, y = batch
            true_values.append(y.numpy())
        true_values = np.concatenate(true_values, axis=0)
        predictions = arima_model.predict(
            test_loader, seq_len, pred_len, feature_columns)
        mse, mae, rmse = arima_model.evaluate(
            true_values, predictions, loss_save_path)
    elif model_name == "casual":
        checkpoint_path = os.path.join(root_dir, f"{dataset_name}/")
        casual_run = CasualTransformer_train_test(
            train_csv=train_data_path,
            test_csv=test_data_path,
            feature_columns=feature_columns,
            seq_len=seq_len,
            pred_len=None,
            max_epochs=40,
            ckpt_path=checkpoint_path,
        )
        casual_run.main()
    else:
        raise ("model name wrong, must be gbrt or arima")

    if model_name != "casual":
        print("\n最终评估结果：")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")


if __name__ == "__main__":
    model_name = "gbrt"         # choose from ( casual || arima || gbrt )
    # choose from (electricity || exchange_rate || PSM || traffic || weather)
    dataset_name = "traffic"
    seq_len, pred_len = 64, 64
    run(model_name, dataset_name, seq_len, pred_len)
    print(f"model_name= {model_name}")
    print(f"dataset_name= {dataset_name}")
    print(f"seq_len-pred_len: {seq_len}_{pred_len}")
