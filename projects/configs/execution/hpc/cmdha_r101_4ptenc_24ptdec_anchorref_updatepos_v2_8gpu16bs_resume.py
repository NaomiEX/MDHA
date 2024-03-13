_base_ = [
    "../../runtime_settings/8gpu16bs_run_r101.py",
    "../../settings/cmdha_r101_4ptenc_24ptdec_anchorref_updatepos_v2.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'PETRTransformerDecoder', 'StreamPETRHead']

resume_from="./work_dirs/cmdha_r101_4ptenc_24ptdec_anchorref_updatepos_v2_8gpu16bs/latest.pth"