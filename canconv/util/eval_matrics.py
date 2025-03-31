import numpy as np
import torch
from scipy.ndimage import sobel
from numpy.linalg import norm
from skimage import filters
from skimage.metrics import peak_signal_noise_ratio as psnr_func

def sam(ms, ps):
    assert ms.ndim == 3 and ms.shape == ps.shape

    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)

    dot_sum = np.sum(ms*ps, axis=2)
    norm_true = norm(ms, axis=2)
    norm_pred = norm(ps, axis=2)

    epsilon = 1e-8  # 小的常数，避免除以零
    res = np.arccos(dot_sum / (norm_pred + epsilon) / (norm_true + epsilon))

    is_nan = np.nonzero(np.isnan(res))

    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0

    sam = np.mean(res)

    return sam * 180 / np.pi


def sCC(ms, ps):
    # ps_sobel = sobel(ps, mode='constant')
    # ms_sobel = sobel(ms, mode='constant')
    ps_sobel = np.zeros(ps.shape)
    ms_sobel = np.zeros(ps.shape)
    for i in range(ms.shape[2]):
        ps_sobel[:, :, i] = filters.sobel(ps[:, :, i])
        ms_sobel[:, :, i] = filters.sobel(ms[:, :, i])

    scc = np.sum(ps_sobel * ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel))

    return scc


def ergas(ms, ps, ratio=8):
    # 将输入的多光谱图像（ms）和配准后的图像（ps）转换为float32类型，以进行精确计算
    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)

    # 计算两个图像之间的差异
    err = ms - ps

    # 初始化ERGAS指数为0
    ergas_index = 0

    # 遍历图像的每个波段
    for i in range(err.shape[2]):
        # 计算当前波段的均方误差（MSE），并除以该波段图像平均值的平方
        ergas_index += np.mean(np.square(err[:, :, i])) / np.square(np.mean(ms[:, :, i]))

    # 计算ERGAS指数的最终值
    # 这里使用了100/ratio，这可能是用来将误差标准化到特定的比例或分辨率
    # 然后乘以1/err.shape[2]的平方根，这是波段数量的倒数的平方根
    # 最后乘以ergas_index，这是所有波段误差的累加和
    ergas_index = (100 / ratio) * np.sqrt(1 / err.shape[2]) * ergas_index

    # 返回计算得到的ERGAS指数
    return ergas_index


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 计算MSE（均方误差）
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr_value

