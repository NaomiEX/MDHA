_base_ = [
    "../../runtime_settings/1gpu2bs_run.py",
    "../../settings/cmdha_24ptenc_4ptdec_anchorref_updatepos_convdepth_mult_newproj.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']