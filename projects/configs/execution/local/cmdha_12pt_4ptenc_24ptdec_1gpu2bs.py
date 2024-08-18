_base_ = [
    "../../runtime_settings/1gpu2bs_run.py",
    "../../settings/nocp/cmdha_12pt_4ptenc_24ptdec_nocp.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'AnchorEncoder']