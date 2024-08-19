_base_ = [
    "../../runtime_settings/2gpu2bs_run.py",
    "../../settings/cmdha_fixed_4ptenc_24ptdec.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'AnchorEncoder']