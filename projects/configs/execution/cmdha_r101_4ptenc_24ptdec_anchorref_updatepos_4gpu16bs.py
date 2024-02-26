_base_ = [
    "../runtime_settings/4gpu16bs_run.py",
    "../settings/cmdha_r101_4ptenc_24ptdec_anchorref_updatepos.py",
    "../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'PETRTransformerDecoder']