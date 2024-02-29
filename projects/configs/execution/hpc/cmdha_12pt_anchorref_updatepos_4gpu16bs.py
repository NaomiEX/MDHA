_base_ = [
    "../runtime_settings/4gpu16bs_run.py",
    "../settings/nocp/cmdha_12pt_anchorref_updatepos_nocp.py",
    "../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'PETRTransformerDecoder']