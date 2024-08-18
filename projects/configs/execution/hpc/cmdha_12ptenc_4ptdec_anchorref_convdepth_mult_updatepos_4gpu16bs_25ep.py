_base_ = [
    "../../runtime_settings/4gpu16bs_run_25ep.py",
    "../../settings/nocp/cmdha_12ptenc_4ptdec_anchorref_updatepos_convdepth_mult.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CircularDeformAttn', 'Projections', 'AnchorEncoder',
               'AnchorRefinement', 'ReferencePoints']