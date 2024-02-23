X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in qulity
YAW = 6  # decoded

# format: [B, N, h0*w0 + h1*w1 + ..., ...]
MLVL_N_HW = 0
# format: [B, h0*N*w0 + h1*N*w1 + ..., ...]
MLVL_HNW = 1

# considers each h_i*w_i as a separate level
FLAT_LVL_HW = 0
# considers each h_i*n*w_i as a separate level
FLAT_LVL_HNW = 1

PC_RANGE=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]