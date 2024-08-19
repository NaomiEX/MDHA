_base_ = [
    "../../runtime_settings/2gpu2bs_run.py",
    "../../settings/cmdha_24ptenc_4ptdec_anchorref_updatepos_convdepth_mult_newproj.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CircularDeformAttn', 'Projections', 'AnchorEncoder',
               'AnchorRefinement', 'ReferencePoints']