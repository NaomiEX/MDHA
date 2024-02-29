_base_ = [
    "../runtime_settings/1gpu2bs_run.py",
    "../settings/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult_nopos3d_nosa.py",
    "../runtime_settings/debug.py"
]

debug_modules=['CircularDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']