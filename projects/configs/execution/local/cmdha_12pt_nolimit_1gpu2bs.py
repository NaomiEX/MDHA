_base_ = [
    "../../runtime_settings/1gpu2bs_run.py",
    "../../settings/cmdha_12pt_nolimit.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'AnchorEncoder']