_base_ = [
    "../runtime_settings/4gpu16bs_run.py",
    "../settings/nocp/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult_dwlim0p4_nocp.py",
    "../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints', 'DepthNet']