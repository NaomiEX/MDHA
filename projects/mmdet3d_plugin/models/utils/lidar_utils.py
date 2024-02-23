import torch
def normalize_lidar(x, pc_range):
    # [pc_min, pc_max] -> [0,1]
    assert x.shape[-1] >= 3
    x[..., 0:3] = (x[..., 0:3] - pc_range[0:3]) / (pc_range[3:6] - pc_range[0:3])
    return x

def denormalize_lidar(x, pc_range):
    # [0,1] -> [pc_min, pc_max]
    assert x.shape[-1] >= 3
    x[..., 0:3] = x[..., 0:3] * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    return x

def clamp_to_lidar_range(x, pc_range):
    assert x.size(-1) >= 3
    x[..., 0] = torch.clamp(x[..., 0].clone(), min=pc_range[0], max=pc_range[3])
    x[..., 1] = torch.clamp(x[..., 1].clone(), min=pc_range[1], max=pc_range[4])
    x[..., 2] = torch.clamp(x[..., 2].clone(), min=pc_range[2], max=pc_range[5])
    return x

def not_in_lidar_range(x, pc_range):
    mask = (x[..., 0] < pc_range[0]) | (x[..., 0] > pc_range[3]) | \
        (x[..., 1] < pc_range[1]) | (x[..., 1] > pc_range[4]) | \
        (x[..., 2] < pc_range[2]) | (x[..., 2] > pc_range[5])
    return mask

def in_lidar_range(x, pc_range):
    return ~not_in_lidar_range(x, pc_range)
