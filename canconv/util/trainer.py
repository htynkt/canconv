# 导入抽象基类相关的模块
from abc import ABCMeta, abstractmethod
# 导入os模块，用于操作系统相关的功能
import os
# 导入glob模块，用于文件路径匹配
from glob import glob
# 导入time模块，用于时间相关的操作
import time
# 导入shutil模块，用于文件操作
import shutil
# 导入logging模块，用于日志记录
import logging
# 导入datetime模块，用于日期和时间的操作
from datetime import datetime
# 导入json模块，用于处理JSON数据
import json
# 导入inspect模块，用于获取对象的信息
import inspect
# 导入torch相关的模块，用于深度学习
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
# 导入tqdm模块，用于显示进度条
from tqdm import tqdm

# 导入自定义的seed模块，用于设置随机种子
from .seed import seed_everything
# 导入自定义的git模块，用于获取git信息
from .git import git, get_git_commit
# 导入自定义的日志模块
from .log import BufferedReporter, to_rgb
# 导入自定义的H5PanDataset类
from ..dataset.h5pan import H5PanDataset
#改：
import torch.nn.functional as F
from eval_matrics import *

# 定义一个抽象基类SimplePanTrainer
class SimplePanTrainer(metaclass=ABCMeta):
    # 定义类属性cfg，用于存储配置信息
    cfg: dict

    # 定义类属性model，用于存储模型
    model: torch.nn.Module
    # 定义类属性criterion，用于存储损失函数
    criterion: torch.nn.Module
    # 定义类属性optimizer，用于存储优化器
    optimizer: torch.optim.Optimizer
    # 定义类属性scheduler，用于存储学习率调度器
    scheduler: torch.optim.lr_scheduler.LRScheduler

    # 定义类属性train_dataset、val_dataset、test_dataset，分别用于存储训练集、验证集和测试集
    train_dataset: H5PanDataset
    val_dataset: H5PanDataset
    test_dataset: H5PanDataset

    # 定义类属性train_loader、val_loader，分别用于存储训练集和验证集的数据加载器
    train_loader: DataLoader
    val_loader: DataLoader

    # 定义类属性out_dir，用于存储输出目录
    out_dir: str

    # 定义类属性disable_alloc_cache，用于控制是否禁用缓存分配
    disable_alloc_cache: bool

    # 定义一个抽象方法forward，用于前向传播，具体实现由子类提供
    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    # 定义一个抽象方法_create_model，用于创建模型，具体实现由子类提供
    @abstractmethod
    def _create_model(self, cfg):
        raise NotImplementedError

    # 定义类的初始化方法
    def __init__(self, cfg):
        # 初始化配置信息
        self.cfg = cfg
        # 初始化日志记录器
        self.logger = logging.getLogger(f"canconv.{cfg['exp_name']}")
        self.logger.setLevel(logging.INFO)
        # 设置随机种子
        seed_everything(cfg["seed"])
        self.logger.info(f"Seed set to {cfg['seed']}")

        # 设置设备
        self.dev = torch.device(cfg['device'])
        # 检查设备是否为cuda
        if self.dev.type != "cuda":
            raise ValueError(f"Only cuda device is supported, got {self.dev.type}")
        # 检查是否使用了非cuda:0的设备
        if self.dev.index != 0:
            self.logger.warning(
                "Warning: Multi-GPU is not supported, the code may not work properly with GPU other than cuda:0. Please use CUDA_VISIBLE_DEVICES to select the device.")
            torch.cuda.set_device(self.dev)

        # 记录使用的设备
        self.logger.info(f"Using device: {self.dev}")
        # 创建模型
        self._create_model(cfg)
        # 测试模型的前向传播
        self.forward({
            'gt': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'ms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 16, 16),
            'lms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'pan': torch.randn(cfg['batch_size'], 1, 64, 64)
        })
        # 设置是否禁用缓存分配
        self.disable_alloc_cache = cfg.get("disable_alloc_cache", False)
        # 记录模型加载完成
        self.logger.info(f"Model loaded.")

    # 定义一个方法用于加载数据集
    def _load_dataset(self):
        # 加载训练集、验证集和测试集
        self.train_dataset = H5PanDataset("wv3/train_wv3.h5")
        self.val_dataset = H5PanDataset("wv3/valid_wv3.h5")
        self.test_dataset = H5PanDataset("wv3/test_wv3_multiExm1.h5")

    # 定义一个方法用于创建输出目录
    def _create_output_dir(self):
        # 创建输出目录
        self.out_dir = os.path.join('runs', self.cfg["exp_name"])
        os.makedirs(os.path.join(self.out_dir, 'weights'), exist_ok=True)
        # 记录输出目录
        logging.info(f"Output dir: {self.out_dir}")

    # 定义一个方法用于保存配置信息
    def _dump_config(self):
        # 保存配置信息到文件
        with open(os.path.join(self.out_dir, "cfg.json"), "w") as file:
            self.cfg["git_commit"] = get_git_commit()
            self.cfg["run_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
            json.dump(self.cfg, file, indent=4)

        # 尝试复制源代码到输出目录
        try:
            source_path = inspect.getsourcefile(self.__class__)
            assert source_path is not None
            source_path = os.path.dirname(source_path)
            shutil.copytree(source_path, os.path.join(self.out_dir, "source"),
                            ignore=shutil.ignore_patterns('*.pyc', '__pycache__'), dirs_exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to copy source code: ")
            self.logger.exception(e)

    # 定义一个方法，训练开始时调用
    def _on_train_start(self):
        pass

    # 定义一个方法，验证开始时调用
    def _on_val_start(self):
        pass

    # 定义一个方法，每个epoch开始时调用
    def _on_epoch_start(self, epoch):
        pass

    # 定义一个方法用于运行测试
    @torch.no_grad()
    def run_test(self, dataset: H5PanDataset):
        # 将模型设置为评估模式
        self.model.eval()
        # 初始化超分辨率图像
        sr = torch.zeros(
            dataset.lms.shape[0], dataset.lms.shape[1], dataset.pan.shape[2], dataset.pan.shape[3], device=self.dev)
        # 对数据集中的每个样本进行前向传播
        for i in range(len(dataset)):
            sr[i:i + 1] = self.forward(dataset[i:i + 1])
        # 返回超分辨率图像
        return sr

    # 定义一个方法用于运行测试，指定图像ID
    @torch.no_grad()
    def run_test_for_selected_image(self, dataset, image_ids):
        # 将模型设置为评估模式
        self.model.eval()
        # 初始化超分辨率图像
        sr = torch.zeros(
            len(image_ids), dataset.lms.shape[1], dataset.pan.shape[2], dataset.pan.shape[3], device=self.dev)
        # 对指定的图像进行前向传播
        for i, image_id in enumerate(image_ids):
            sr[i:i + 1] = self.forward(dataset[image_id:image_id + 1])
        # 返回超分辨率图像
        return sr

    # 定义训练方法
    def train(self):
        # 加载数据集
        self._load_dataset()
        # 创建训练集的数据加载器
        train_loader = DataLoader(
            dataset=self.train_dataset,  # 指定训练集
            batch_size=self.cfg['batch_size'],  # 设置每个批次的大小
            shuffle=True,  # 打乱数据顺序
            drop_last=False,  # 不丢弃最后一个不完整的批次
            pin_memory=True)  # 使用pin_memory加速数据传输
        # 创建验证集的数据加载器
        val_loader = DataLoader(
            dataset=self.val_dataset,  # 指定验证集
            batch_size=self.cfg['batch_size'],  # 设置每个批次的大小
            shuffle=True,  # 打乱数据顺序
            drop_last=False,  # 不丢弃最后一个不完整的批次
            pin_memory=True)  # 使用pin_memory加速数据传输
        # 记录数据集加载完成
        self.logger.info(f"Dataset loaded.")

        # 创建输出目录
        self._create_output_dir()
        # 保存配置信息
        self._dump_config()
        # 调用训练开始时的方法
        self._on_train_start()

        # 创建TensorBoard的SummaryWriter，用于记录训练过程
        writer = SummaryWriter(log_dir=self.out_dir)
        # 初始化训练损失的记录器
        train_loss = BufferedReporter(f'train/{self.criterion.__class__.__name__}', writer)
        # 初始化验证损失的记录器
        val_loss = BufferedReporter(f'val/{self.criterion.__class__.__name__}', writer)
        # 初始化训练时间的记录器
        train_time = BufferedReporter('train/time', writer)
        # 初始化验证时间的记录器
        val_time = BufferedReporter('val/time', writer)
        #改：
        sam_data = BufferedReporter('Metrics/sam', writer)
        scc_data = BufferedReporter('Metrics/scc', writer)

        # 记录开始训练
        self.logger.info(f"Begin Training.")

        # 遍历每个epoch
        for epoch in tqdm(range(1, self.cfg['epochs'] + 1, 1)):
            # 调用每个epoch开始时的方法
            self._on_epoch_start(epoch)

            # 将模型设置为训练模式
            self.model.train()
            # 遍历训练集的每个批次
            for batch in tqdm(train_loader):
                # 记录开始时间
                start_time = time.time()

                # 清空模型的梯度
                self.model.zero_grad()
                # 前向传播，得到超分辨率图像
                sr = self.forward(batch)

                #改：
                B_hat = self.kenet(batch['lms'].to(self.dev))
                B_hat = B_hat.unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, 7, 7]
                B_hat = B_hat.repeat(self.cfg['spectral_num'],self.cfg['spectral_num'] , 1, 1)  # 形状变为 [4, 4, 7, 7]
                ms_eval = F.conv2d(sr, B_hat, padding=3)

                fused_eval=self.Adata * sr
                # 计算损失
                loss1 = self.criterion1(sr, ms_eval.to(self.dev)) #?在dev上吗
                loss2 = self.criterion2(sr, fused_eval.to(self.dev)) #?在dev上吗
                loss=loss1+loss2

                # 将损失值添加到训练损失的记录器中
                train_loss.add_scalar(loss.item())

                #改：(检查归一化)
                sam_data.add_scalar(sam(sr[0].cpu().detach().numpy(), batch['lms'][0].cpu().detach().numpy()))
                scc_data.add_scalar(sCC(sr[0].cpu().detach().numpy(), batch['lms'][0].cpu().detach().numpy()))

                # 反向传播
                loss.backward()
                # 优化器更新参数
                self.optimizer.step()

                # 如果禁用了缓存分配，则清空缓存
                if self.disable_alloc_cache:
                    torch.cuda.empty_cache()

                # 将训练时间添加到记录器中
                train_time.add_scalar(time.time() - start_time)
            # 将训练损失和时间的记录器刷新到TensorBoard
            train_loss.flush(epoch)
            train_time.flush(epoch)

            #改：
            sam_data.flush(epoch)
            scc_data.flush(epoch)

            # 学习率调度器更新学习率
            self.scheduler.step()
            # 记录当前epoch的训练完成
            self.logger.debug(f"Epoch {epoch} train done")

            #验证？还需要吗
            # 如果当前epoch是验证间隔的整数倍
            if epoch % self.cfg['val_interval'] == 0:
                # 调用验证开始时的方法
                self._on_val_start()
                # 将模型设置为评估模式
                with torch.no_grad():
                    self.model.eval()
                    # 遍历验证集的每个批次
                    for batch in val_loader:
                        # 记录开始时间
                        start_time = time.time()
                        # 前向传播，得到超分辨率图像
                        sr = self.forward(batch)
                        # 计算损失
                        loss = self.criterion(sr, batch['gt'].to(self.dev))
                        # 将损失值添加到验证损失的记录器中
                        val_loss.add_scalar(loss.item())
                        # 将验证时间添加到记录器中
                        val_time.add_scalar(time.time() - start_time)
                    # 将验证损失和时间的记录器刷新到TensorBoard
                    val_loss.flush(epoch)
                    val_time.flush(epoch)
                # 记录当前epoch的验证完成
                self.logger.debug(f"Epoch {epoch} val done")

            # 如果当前epoch是检查点间隔的整数倍，或者配置中指定了保存第一个epoch
            if epoch % self.cfg['checkpoint'] == 0 or (
                    "save_first_epoch" in self.cfg and epoch <= self.cfg["save_first_epoch"]):
                # 保存模型的参数到文件
                torch.save(self.model.state_dict(), os.path.join(
                    self.out_dir, f'weights/{epoch}.pth'))
                # 记录检查点保存完成
                self.logger.info(f"Epoch {epoch} checkpoint saved")

        # 保存最终的模型参数到文件
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "weights/final.pth"))
        # 记录训练完成
        self.logger.info(f"Training finished.")