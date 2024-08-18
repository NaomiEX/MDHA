_base_ = [
    "../../runtime_settings/2gpu4bs_run.py",
    "../../settings/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult_encanchorpos.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'AnchorEncoder',
               'AnchorRefinement', 'ReferencePoints']