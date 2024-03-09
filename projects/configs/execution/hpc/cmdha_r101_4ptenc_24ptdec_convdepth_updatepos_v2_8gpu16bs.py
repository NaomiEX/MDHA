_base_ = [
    "../../runtime_settings/8gpu16bs_run_r101.py",
    "../../settings/cmdha_r101_4ptenc_24ptdec_convdepth_updatepos_v2.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'PETRTransformerDecoder', 'DepthNet',
               'StreamPETRHead']