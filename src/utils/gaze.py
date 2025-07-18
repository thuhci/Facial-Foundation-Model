import torch
from typing import Dict, Any
from torch.utils.data import DataLoader
from src.utils.config import get_cfg

def spherical_to_cartesian(spherical_coords):
    """将球坐标转换为笛卡尔坐标"""
    pitch = spherical_coords[:, 0]  # 俯仰角
    yaw = spherical_coords[:, 1]    # 偏航角
    
    # 转换为笛卡尔坐标
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    
    return torch.stack([x, y, z], dim=1)

def cartesian_to_spherical(cartesian_coords):
    """将笛卡尔坐标转换为球坐标"""
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]
    
    # 计算俯仰角和偏航角
    pitch = torch.asin(torch.clamp(y, -1.0, 1.0))
    yaw = torch.atan2(x, -z)
    
    return torch.stack([pitch, yaw], dim=1)

def compute_angular_error(pred_spherical, target_spherical):
    """计算角度误差"""
    # 转换为笛卡尔坐标
    pred_cartesian = spherical_to_cartesian(pred_spherical)
    target_cartesian = spherical_to_cartesian(target_spherical)
    
    # 计算点积
    dot_product = torch.sum(pred_cartesian * target_cartesian, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # 计算角度误差（弧度转角度）
    angular_error = torch.acos(dot_product) * 180.0 / torch.pi
    return torch.mean(angular_error)

def gaze3d_to_gaze2d(gaze_3d):
    """将3D gaze向量转换为2D角度（pitch, yaw）"""
    x = gaze_3d[:, 0]
    y = gaze_3d[:, 1]
    z = gaze_3d[:, 2]
    
    pitch = torch.asin(torch.clamp(y, -1.0, 1.0))
    yaw = torch.atan2(x, -z)
    
    # pitch = pitch* 180 / torch.pi
    # yaw = yaw* 180 / torch.pi
    
    return torch.stack([pitch, yaw], dim=1)

def angles_to_bins(angles, num_bins=90, bin_width=2.0):
    """将角度转换为分类标签"""
    # 角度范围：-90 到 90 度
    angles_deg = angles * 180.0 / torch.pi
    # angles_deg = angles
    bins = torch.floor((angles_deg + 90.0) / bin_width).long()
    bins = torch.clamp(bins, 0, num_bins - 1)
    return bins

def bins_to_angles(bins, idx_tensor, bin_width=2.0):
    """将分类结果转换为角度"""
    angles_deg = torch.sum(bins * idx_tensor, 1) * bin_width - 90.0
    return angles_deg * torch.pi / 180.0
    # return angles_deg

def l2cs_loss(pitch_pred, yaw_pred, pitch_target, yaw_target, 
              pitch_bins, yaw_bins, idx_tensor, 
              criterion_ce, criterion_mse, alpha=1.0):
    """L2CS损失函数"""
    # 分类损失
    loss_pitch_ce = criterion_ce(pitch_pred, pitch_bins)
    loss_yaw_ce = criterion_ce(yaw_pred, yaw_bins)
    
    # 回归损失
    softmax = torch.nn.Softmax(dim=1)
    pitch_predicted = softmax(pitch_pred)
    yaw_predicted = softmax(yaw_pred)
    
    pitch_continuous = bins_to_angles(pitch_predicted, idx_tensor)
    yaw_continuous = bins_to_angles(yaw_predicted, idx_tensor)
    
    loss_pitch_mse = criterion_mse(pitch_continuous, pitch_target)
    loss_yaw_mse = criterion_mse(yaw_continuous, yaw_target)
    
    # 总损失
    total_loss = loss_pitch_ce + loss_yaw_ce + alpha * (loss_pitch_mse + loss_yaw_mse)
    
    # print("pitch_predicted:", pitch_predicted, "yaw_predicted:", yaw_predicted)
    # print("pitch_continuous:", pitch_continuous, "yaw_continuous:", yaw_continuous)
    # print("pitch_target:", pitch_target, "yaw_target:", yaw_target)
    # print("pitch_bins:", pitch_bins, "yaw_bins:", yaw_bins)
    # print("loss_pitch_ce:", loss_pitch_ce.item(), "loss_yaw_ce:", loss_yaw_ce.item())
    # print("loss_pitch_mse:", loss_pitch_mse.item(), "loss_yaw_mse:", loss_yaw_mse.item())
    # print("total_loss:", total_loss.item())
        
    # 计算角度误差
    pred_angles = torch.stack([pitch_continuous, yaw_continuous], dim=1)
    target_angles = torch.stack([pitch_target, yaw_target], dim=1)
    angular_error = compute_angular_error(pred_angles, target_angles)
    
    return total_loss, loss_pitch_ce + loss_yaw_ce, loss_pitch_mse + loss_yaw_mse, angular_error


def criterion_l2cs(outputs, targets):
    cfg = get_cfg()
            # 假设 targets 是 3D gaze 向量
    gaze_2d = gaze3d_to_gaze2d(targets)
    pitch_target = gaze_2d[:, 0]
    yaw_target = gaze_2d[:, 1]
    
    # 转换为分类标签
    pitch_bins = angles_to_bins(pitch_target, cfg.GAZE.NUM_BINS, cfg.GAZE.BIN_WIDTH)
    yaw_bins = angles_to_bins(yaw_target, cfg.GAZE.NUM_BINS, cfg.GAZE.BIN_WIDTH)
    
    # 模型输出
    pitch_pred = outputs['pitch']
    yaw_pred = outputs['yaw']
    
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()
    idx_tensor = torch.arange(0, cfg.GAZE.NUM_BINS, device=pitch_bins.device).float()
    
    return l2cs_loss(pitch_pred, yaw_pred, pitch_target, yaw_target,
                    pitch_bins, yaw_bins, idx_tensor,
                    criterion_ce, criterion_mse, cfg.GAZE.ALPHA_REG)
    
    