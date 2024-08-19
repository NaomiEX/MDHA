_base_ = [
    "../../runtime_settings/4gpu16bs_run.py",
    "../../settings/nocp/cmdha_fixed_4ptenc_24ptdec_nocp.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'AnchorEncoder']