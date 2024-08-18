# Modified from Sparse DETR

import torch

def attn_map_to_flat_grid_multiple(spatial_shapes, level_start_index, 
                          sampling_locations, attention_weights):
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
    # attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, _, n_heads, *_ = sampling_locations.shape
    # [N, n_layers, n_heads, Q, n_points, n_levels, 2]
    # [N * n_layers * n_heads, Len_q * n_points, n_levels, 2]
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2)
    # hw -> wh (xy)
    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) 
    # sampling_locations are normalized in range [0,1] so here we unnormalize it by multiplying with (w, h)
    # to get the (x,y) sampling positions
    col_row_float = sampling_locations * rev_spatial_shapes # [N * n_layers * n_heads, Len_q * n_points, n_levels, 2]
    # take the floor to convert to int (x,y) coords
    col_row_ll = col_row_float.floor().to(torch.int64)

    # [N * n_layers * n_heads, Len_q * n_points, n_levels]
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    
    # add 0 to every x, add 1 to every y
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)
    # add 1 to every x, add 1 to every y
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)
    # add 1 to every x and y
    col_row_hh = col_row_ll + 1

    # gets the difference between the float coords and the floor (truncated) ones and take the product (diff_x * diff_y)
    margin_ll = (col_row_float - col_row_ll).prod(dim=-1) # [N * n_layers * n_heads, Len_q * n_points, n_levels]
    # take the product: x-x_lower * y-y_upper, and - to get the absolute value
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    # take the product: x-x_upper * y-y_lower, and - to get the absolute value
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    # take the product: x-x_upper * y-y_upper
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)

    # 2nd dim is for the attention grid over the flattened input img feats
    # [N * n_layers * n_heads, H_0*W_0+H_1*W_1+...]
    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        # mask where True means the (x,y) are within range [0, w] and [0, h], False otherwise
        # [N * n_layers * n_heads, Len_q * n_points, n_levels]
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )
        # get the flattened idx
        idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
        # 0 out invalid values
        idx = (idx * valid_mask).flatten(1, 2) # [N * n_layers * n_heads, Len_q * n_points *n_levels]
        # multiply the attention weights by the margin,
        # [N * n_layers * n_heads,   Len_q * n_points * n_levels]
        # NOTE: col_row and margin pair ups are opposites because, 
        #       for ex., if margin_hh is large, margin_ll is small because margin_ll is 1-margin_hh
        #       the smaller the margin, the more relevant, which is why we multiply with margin_hh instead of margin_ll
        # NOTE: the margin factor all add up to 1 so it wont go beyond 1
        weights = (attention_weights * valid_mask * margin).flatten(1)
        # add the modified attention weights into the flat_grid based on idx
        flat_grid.scatter_add_(1, idx, weights)

    # [N, n_layers, n_heads, H_0*W_0+H_1*W_1+...]
    return flat_grid.reshape(N, n_layers, n_heads, -1)

def attn_map_to_flat_grid_single(spatial_shapes, level_start_index, 
                                sampling_locations, attention_weights):
    """
    Args:
        spatial_shapes (_type_): _description_
        level_start_index (_type_): _description_
        sampling_locations (_type_): [B, Q, n_heads, n_feature_lvls, n_points, 2]
        attention_weights (_type_): [B, Q, n_heads, n_feature_lvls, n_points]
    """
    N, Q, n_heads, *_ = sampling_locations.shape
    # [N, n_heads, Q, n_points, n_feature_lvls, 2] -> [N*n_heads, Q*n_points, n_levels, 2]
    sampling_locations = sampling_locations.permute(0, 2, 1, 4, 3, 5).flatten(0, 1).flatten(1, 2)
    # [N, n_heads, Q, n_points, n_feature_lvls] -> [N*n_heads, Q*n_points, n_levels]
    attention_weights = attention_weights.permute(0, 2, 1, 4, 3).flatten(0,1).flatten(1, 2)
    # hw -> wh (xy)
    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) # [n_levels, 2] 
    # sampling_locations are normalized in range [0,1] so here we unnormalize it by multiplying with (w, h)
    # to get the (x,y) sampling positions
    col_row_float = sampling_locations * rev_spatial_shapes # [N*n_heads, Q*n_points, n_levels, 2]
    # take the floor to convert to int (x,y) coords
    col_row_ll = col_row_float.floor().to(torch.int64)
    # [N * n_heads, Len_q * n_points, n_levels]
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    
    # add 0 to every x, add 1 to every y
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)
    # add 1 to every x, add 1 to every y
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)
    # add 1 to every x and y
    col_row_hh = col_row_ll + 1

    # gets the difference between the float coords and the floor (truncated) ones and take the product (diff_x * diff_y)
    margin_ll = (col_row_float - col_row_ll).prod(dim=-1) # [N * n_heads, Len_q * n_points, n_levels]
    # take the product: x-x_lower * y-y_upper, and - to get the absolute value
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    # take the product: x-x_upper * y-y_lower, and - to get the absolute value
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    # take the product: x-x_upper * y-y_upper
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)

    # 2nd dim is for the attention grid over the flattened input img feats
    # [N * n_heads, H_0*W_0+H_1*W_1+...]
    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        # mask where True means the (x,y) are within range [0, w] and [0, h], False otherwise
        # [N * n_heads, Len_q * n_points, n_levels]
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )
        # get the flattened idx
        # [N*n_heads, Q*n_points, n_levels] * W + [N*n_heads, Q*n_points, n_levels] + [n_levels]
        # y * W + x + level_start
        idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
        # 0 out invalid values
        idx = (idx * valid_mask).flatten(1, 2) # [N * n_heads, Len_q * n_points *n_levels]
        # multiply the attention weights by the margin,
        # [N * n_layers * n_heads,   Len_q * n_points * n_levels]
        # NOTE: col_row and margin pair ups are opposites because, 
        #       for ex., if margin_hh is large, margin_ll is small because margin_ll is 1-margin_hh
        #       the smaller the margin, the more relevant, which is why we multiply with margin_hh instead of margin_ll
        # NOTE: the margin factor all add up to 1 so it wont go beyond 1
        weights = (attention_weights * valid_mask * margin).flatten(1) # [N * n_heads, Len_q * n_points * n_levels]
        # add the modified attention weights into the flat_grid based on idx
        flat_grid.scatter_add_(1, idx, weights)
    return flat_grid.reshape(N, n_heads, -1) # [N, n_heads, h0*N*w0+...]

def attn_map_to_flat_grid(spatial_shapes, level_start_index, 
                          sampling_locations, attention_weights,
                          sum_out=True):
    """
    Args:
        spatial_shapes (_type_): [n_levels, 2] (H, W*num_cameras)
        level_start_index (_type_): [n_levels]
        sampling_locations (_type_): [B, num_dec_layers, Q, n_heads, n_levels, n_points, 2]
        attention_weights (_type_): [B, num_dec_layers, Q, n_heads, n_levels, n_points]

    Returns:
        _type_: _description_
    """
    if sampling_locations.dim() == 7 and attention_weights.dim() == 6:
        flat_attn_map = attn_map_to_flat_grid_multiple(spatial_shapes, level_start_index, 
                          sampling_locations, attention_weights)
        if sum_out:
            flat_attn_map = flat_attn_map.sum(dim=(1, 2))
    elif sampling_locations.dim()==6 and attention_weights.dim() == 5:
        flat_attn_map = attn_map_to_flat_grid_single(spatial_shapes, level_start_index, 
                          sampling_locations, attention_weights)
        if sum_out:
            flat_attn_map = flat_attn_map.sum(dim=1)
    else:
        raise ValueError(f"got unexpected combo sampling_locations.dim={sampling_locations.dim()}"\
                         f" and attention_weights.dim={attention_weights.dim()}")
    if sum_out:
        assert flat_attn_map.dim() == 2 # [B, h0*N*w0+...]
    return flat_attn_map