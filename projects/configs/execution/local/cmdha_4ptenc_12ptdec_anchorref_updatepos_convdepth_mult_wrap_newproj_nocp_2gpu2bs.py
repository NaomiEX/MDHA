_base_ = [
    "../../runtime_settings/2gpu2bs_run.py",
    "../../settings/nocp/cmdha_4ptenc_12ptdec_anchorref_updatepos_convdepth_mult_wrap_newproj_nocp.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CircularDeformAttn']