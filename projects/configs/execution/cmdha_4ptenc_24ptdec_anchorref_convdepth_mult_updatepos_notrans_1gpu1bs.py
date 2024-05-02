_base_ = [
    "../runtime_settings/1gpu1bs_run_1ep.py",
    "../settings/nocp/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult_notrans.py",
    "../runtime_settings/debug.py"
]

debug_modules=['CircularDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']