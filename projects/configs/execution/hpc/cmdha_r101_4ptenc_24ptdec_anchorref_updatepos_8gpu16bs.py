_base_ = [
    "../../runtime_settings/8gpu16bs_run_r101.py",
    "../../settings/cmdha_r101_4ptenc_24ptdec_anchorref_updatepos.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'PETRTransformerDecoder', 'StreamPETRHead']