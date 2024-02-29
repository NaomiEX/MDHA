_base_ = [
    "../runtime_settings/2gpu2bs_run.py",
    "../settings/nocp/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult.py",
    "../runtime_settings/debug.py"
]

debug_modules=['CircularDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']