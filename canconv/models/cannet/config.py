# 导入PyTorch相关的模块
import torch
import torch.nn as nn

# 导入自定义的CANNet模型
from .model import CANNet
#改：
from canconv.models.cannet.model import KENet

# 导入SimplePanTrainer抽象基类
from canconv.util.trainer import SimplePanTrainer
# 导入KMeans相关的工具
from canconv.layers.kmeans import reset_cache, KMeansCacheScheduler


# 定义CANNetTrainer类，继承自SimplePanTrainer
class CANNetTrainer(SimplePanTrainer):
    # 定义初始化方法
    def __init__(self, cfg) -> None:
        # 调用父类的初始化方法
        super().__init__(cfg)

    # 定义创建模型的方法
    def _create_model(self, cfg):
        # 根据配置选择损失函数
        if cfg["loss"] == "l1":
            self.criterion1 = nn.L1Loss(reduction='mean').to(self.dev)
            self.criterion2 = nn.L1Loss(reduction='mean').to(self.dev)
        elif cfg["loss"] == "l2":
            self.criterion = nn.MSELoss(reduction='mean').to(self.dev)
        else:
            raise NotImplementedError(f"Loss {cfg['loss']} not implemented")
        # 初始化CANNet模型
        self.model = CANNet(cfg['spectral_num'], cfg['channels'],
                            cfg['cluster_num'], cfg["filter_threshold"]).to(self.dev)
        # 改：
        self.kenet=KENet(spectral_num=cfg['spectral_num'])
        self.Adata=self.model.custom_param
        # 初始化优化器?
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.kenet.parameters(),lr=cfg["learning_rate"], weight_decay=0)

        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg["lr_step_size"])

        # 初始化KMeans缓存调度器
        self.km_scheduler = KMeansCacheScheduler(cfg['kmeans_cache_update'])

    # 定义训练开始时调用的方法
    def _on_train_start(self):
        # 重置KMeans缓存，长度为训练集的大小
        reset_cache(len(self.train_dataset))

    # 定义每个epoch开始时调用的方法
    def _on_epoch_start(self, epoch):
        # 调用KMeans缓存调度器的step方法
        self.km_scheduler.step()

    # 定义前向传播方法
    def forward(self, data):
        # 检查输入数据中是否包含索引
        if "index" in data:
            # 如果包含索引，调用模型的forward方法，并传入索引
            return self.model(data['pan'].to(self.dev), data['lms'].to(self.dev), data['index'].to(self.dev))
        else:
            # 如果不包含索引，调用模型的forward方法，不传入索引
            return self.model(data['pan'].to(self.dev), data['lms'].to(self.dev))