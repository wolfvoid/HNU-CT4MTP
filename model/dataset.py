from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os

class CSVDataset(Dataset):
    """从CSV文件加载时间序列数据集，支持单个文件或文件夹"""

    def __init__(
        self,
        csv_path="data/train_data_processed",
        seq_len=100,
        pred_len=1,
        valid=False,
        feature_columns=None,
    ):
        """
        初始化数据集

        参数:
            csv_path: CSV文件路径或包含CSV文件的文件夹路径
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            valid: 是否使用验证模式（True时使用分段方式而不是滑动窗口）
            feature_columns: 要使用的特征列名列表，例如 ['采集值x', '采集值y', '采集值z']
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.valid = valid
        self.feature_columns = feature_columns
        self.data = []

        # 初始化统计信息
        self.feature_means = None
        self.feature_stds = None

        # 处理输入路径
        if os.path.isfile(csv_path):
            # 单个文件
            self._compute_statistics(csv_path)  # 先计算统计信息
            self.data.extend(self._load_data(csv_path))
        elif os.path.isdir(csv_path):
            # 文件夹
            csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            if not csv_files:
                raise ValueError(f"在 {csv_path} 中没有找到CSV文件")

            # 首先计算所有文件的统计信息
            all_data = []
            for csv_file in csv_files:
                file_path = os.path.join(csv_path, csv_file)
                df = pd.read_csv(file_path)
                if self.feature_columns is None:
                    raise ValueError("必须指定要使用的特征列名列表")

                all_data.append(df[self.feature_columns].values)

            # 计算整体统计信息
            combined_data = np.concatenate(all_data, axis=0)
            self.feature_means = np.mean(combined_data, axis=0)
            self.feature_stds = np.std(combined_data, axis=0)

            # 然后加载数据
            for csv_file in csv_files:
                file_path = os.path.join(csv_path, csv_file)
                self.data.extend(self._load_data(file_path))
        else:
            raise ValueError(f"无效的路径: {csv_path}")

        if not self.data:
            raise ValueError("没有加载到任何有效数据")

        print(f"总共加载了 {len(self.data)} 个样本")
        print(f"特征均值: {self.feature_means}")
        print(f"特征标准差: {self.feature_stds}")

        # example
        print("===========")
        print("第一组数据：")
        print("输入序列：", self.data[0][0])
        print("目标序列：", self.data[0][1])

    def _compute_statistics(self, csv_file):
        """计算特征的均值和标准差"""
        df = pd.read_csv(csv_file)
        if self.feature_columns is None:
            raise ValueError("必须指定要使用的特征列名列表")
        features = df[self.feature_columns].values
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        self.feature_seds = np.maximum(self.feature_stds, 1e-6) #避免标准差为0

    def _load_data(self, csv_file):
        """加载单个CSV文件并处理成序列"""
        # print(f"正在加载文件: {csv_file}")
        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"警告: 无法加载文件 {csv_file}: {str(e)}")
            return []

        # 验证并获取特征列
        if self.feature_columns is None:
            raise ValueError("必须指定要使用的特征列名列表")

        # 检查所有指定的特征列是否存在
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"在文件 {csv_file} 中未找到以下特征列: {missing_cols}")

        # 提取特征数据
        features = df[self.feature_columns].values  # (seq_len, num_features)

        # 创建数据样本
        data = []

        # 检查数据长度是否足够
        if len(df) <= self.seq_len + self.pred_len:
            print(
                f"警告: {csv_file} 的数据长度 ({len(df)}) 小于序列长度+预测长度 ({self.seq_len + self.pred_len})，跳过此文件"
            )
            return data

        if self.valid:
            # 验证模式：将数据分成不重叠的段
            total_samples = (len(df) - self.pred_len) // self.seq_len
            for i in range(total_samples):
                start_idx = i * self.seq_len
                # 输入序列
                input_seq = features[start_idx : start_idx + self.seq_len]
                # 目标序列
                target_seq = features[start_idx + 1 : start_idx + self.seq_len + 1]
                data.append((input_seq, target_seq))
        else:
            # 训练模式：使用滑动窗口
            for i in range(len(df) - self.seq_len - self.pred_len + 1):
                # 输入序列: (seq_len, num_features)
                input_seq = features[i : i + self.seq_len]
                # 目标序列: (seq_len, num_features)
                target_seq = features[i + 1 : i + self.seq_len + 1]
                data.append((input_seq, target_seq))

        # print(f"从 {csv_file} 加载了 {len(data)} 个样本")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class CSVDataset22(Dataset):
    """从CSV文件加载时间序列数据集，支持单个文件或文件夹"""

    def __init__(
        self,
        csv_path="data/train_data_processed",
        seq_len=100,
        pred_len=1,
        valid=False,
        feature_columns=None,
    ):
        """
        初始化数据集

        参数:
            csv_path: CSV文件路径或包含CSV文件的文件夹路径
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            valid: 是否使用验证模式（True时使用分段方式而不是滑动窗口）
            feature_columns: 要使用的特征列名列表，例如 ['采集值x', '采集值y', '采集值z']
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.valid = valid
        self.feature_columns = feature_columns
        self.data = []

        # 初始化统计信息
        self.feature_means = None
        self.feature_stds = None

        # 处理输入路径
        if os.path.isfile(csv_path):
            # 单个文件
            self._compute_statistics(csv_path)  # 先计算统计信息
            self.data.extend(self._load_data(csv_path))
        elif os.path.isdir(csv_path):
            # 文件夹
            csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            if not csv_files:
                raise ValueError(f"在 {csv_path} 中没有找到CSV文件")

            # 首先计算所有文件的统计信息
            all_data = []
            for csv_file in csv_files:
                file_path = os.path.join(csv_path, csv_file)
                df = pd.read_csv(file_path)
                if self.feature_columns is None:
                    raise ValueError("必须指定要使用的特征列名列表")

                all_data.append(df[self.feature_columns].values)

            # 计算整体统计信息
            combined_data = np.concatenate(all_data, axis=0)
            self.feature_means = np.mean(combined_data, axis=0)
            self.feature_stds = np.std(combined_data, axis=0)

            # 然后加载数据
            for csv_file in csv_files:
                file_path = os.path.join(csv_path, csv_file)
                self.data.extend(self._load_data(file_path))
        else:
            raise ValueError(f"无效的路径: {csv_path}")

        if not self.data:
            raise ValueError("没有加载到任何有效数据")

        print(f"总共加载了 {len(self.data)} 个样本")
        print(f"特征均值: {self.feature_means}")
        print(f"特征标准差: {self.feature_stds}")

        # example
        print("===========")
        print("第一组数据：")
        print("输入序列：", self.data[0][0])
        print("目标序列：", self.data[0][1])

    def _compute_statistics(self, csv_file):
        """计算特征的均值和标准差"""
        df = pd.read_csv(csv_file)
        if self.feature_columns is None:
            raise ValueError("必须指定要使用的特征列名列表")
        features = df[self.feature_columns].values
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        self.feature_seds = np.maximum(self.feature_stds, 1e-6) #避免标准差为0

    def _load_data(self, csv_file):
        """加载单个CSV文件并处理成序列"""
        # print(f"正在加载文件: {csv_file}")
        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"警告: 无法加载文件 {csv_file}: {str(e)}")
            return []

        # 验证并获取特征列
        if self.feature_columns is None:
            raise ValueError("必须指定要使用的特征列名列表")

        # 检查所有指定的特征列是否存在
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"在文件 {csv_file} 中未找到以下特征列: {missing_cols}")

        # 提取特征数据
        features = df[self.feature_columns].values  # (seq_len, num_features)

        # 创建数据样本
        data = []

        # 检查数据长度是否足够
        if len(df) <= self.seq_len + self.pred_len:
            print(
                f"警告: {csv_file} 的数据长度 ({len(df)}) 小于序列长度+预测长度 ({self.seq_len + self.pred_len})，跳过此文件"
            )
            return data

        if self.valid:
            # 验证模式：将数据分成不重叠的段
            total_samples = (len(df) - self.pred_len) // self.seq_len
            for i in range(total_samples):
                start_idx = i * self.seq_len
                # 输入序列
                input_seq = features[start_idx : start_idx + self.seq_len]
                # 目标序列
                target_seq = features[start_idx + 1 : start_idx + self.seq_len + 1]
                data.append((input_seq, target_seq))
        else:
            # 训练模式：使用滑动窗口
            for i in range(len(df) - self.seq_len - self.pred_len + 1):
                # 输入序列: (seq_len, num_features)
                input_seq = features[i : i + self.seq_len]
                # 目标序列: (seq_len, num_features)
                target_seq = features[i + 1 : i + self.seq_len + 1]
                data.append((input_seq, target_seq))

        # print(f"从 {csv_file} 加载了 {len(data)} 个样本")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class CSVDataset2(Dataset):
    def __init__(self, csv_path, seq_len, pred_len, valid, is_autoregression=False, feature_columns=None):
        """
        初始化数据集

        参数:
            csv_path: CSV文件路径或包含CSV文件的文件夹路径
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            valid: 是否使用验证模式（True时使用分段方式而不是滑动窗口）
            feature_columns: 要使用的特征列名列表，默认为 None，表示使用所有数值列
        """
        self.is_autoregression = is_autoregression
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.valid = valid
        self.feature_columns = feature_columns  # 特征列名列表
        self.data = []
        self.feature_means = None
        self.feature_stds = None

        print("start loading data...")
        if os.path.isfile(csv_path):
            means, stds, _ = self._load_and_process_file(csv_path)
            self.feature_means, self.feature_stds = means, stds
        elif os.path.isdir(csv_path):
            csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            if not csv_files:
                raise ValueError(f"No CSV files found in {csv_path}")

            file_means = []
            file_stds = []
            file_counts = []

            for csv_file in tqdm(csv_files, desc=f"Loading from {csv_files}"):
                file_path = os.path.join(csv_path, csv_file)
                means, stds, counts = self._load_and_process_file(file_path)
                file_means.append(means)
                file_stds.append(stds)
                file_counts.append(counts)

            # 计算全局均值
            self.feature_means = np.average(file_means, weights=file_counts, axis=0)
            # 计算全局方差
            self.feature_stds = np.sqrt(np.average((np.array(file_stds)**2) + (np.array(file_means) - self.feature_means) ** 2, weights=file_counts, axis=0))

        else:
            raise ValueError(f"Invalid path: {csv_path}")

        if not self.data:
            raise ValueError("No valid data loaded")

        print("finish loading data...")
        return

    def _load_and_process_file(self, file_path):
        """从单个文件中读取数据并生成序列"""
        df = pd.read_csv(file_path)
        if self.feature_columns is None:
            self.feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        features = df[self.feature_columns].values
        file_mean = np.mean(features, axis=0)
        file_std = np.std(features, axis=0, ddof=1)
        file_count = features.shape[0]
        sequences = self._generate_sequences(features)
        self.data.extend(sequences)
        return file_mean, file_std, file_count

    def _generate_sequences(self, features):
        """生成输入序列和目标序列"""
        data = []
        if self.is_autoregression==False:
            #
            if self.valid:
                # 验证模式：将数据分成不重叠的段
                total_samples = (len(features) - self.pred_len) // self.seq_len
                for i in range(total_samples):
                    start_idx = i * self.seq_len
                    input_seq = features[start_idx : start_idx + self.seq_len]
                    target_seq = features[start_idx + self.seq_len: start_idx + self.seq_len + self.pred_len]
                    data.append((input_seq, target_seq))
            else:
                # 训练模式：使用滑动窗口
                for i in range(len(features) - self.seq_len - self.pred_len + 1):
                    input_seq = features[i : i + self.seq_len]
                    target_seq = features[i + self.seq_len : i + self.seq_len + self.pred_len]
                    data.append((input_seq, target_seq))
        else:
            # 基于自回归
            pred_len = 1
            if self.valid:
                # 验证模式：将数据分成不重叠的段
                total_samples = (len(features) - pred_len) // self.seq_len
                for i in range(total_samples):
                    start_idx = i * self.seq_len
                    # 输入序列
                    input_seq = features[start_idx : start_idx + self.seq_len]
                    # 目标序列
                    target_seq = features[start_idx + 1 : start_idx + self.seq_len + 1]
                    data.append((input_seq, target_seq))
            else:
                # 训练模式：使用滑动窗口
                for i in range(len(features) - self.seq_len - pred_len + 1):
                    # 输入序列: (seq_len, num_features)
                    input_seq = features[i : i + self.seq_len]
                    # 目标序列: (seq_len, num_features)
                    target_seq = features[i + 1 : i + self.seq_len + 1]
                    data.append((input_seq, target_seq))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)
