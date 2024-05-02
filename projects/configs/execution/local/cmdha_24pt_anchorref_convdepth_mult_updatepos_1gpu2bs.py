_base_ = [
    "../../runtime_settings/1gpu2bs_run.py",
    "../../settings/nocp/cmdha_24pt_anchorref_updatepos_convdepth_mult.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']