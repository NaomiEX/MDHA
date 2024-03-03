_base_ = [
    "../../runtime_settings/4gpu16bs_run_25ep.py",
    "../../settings/nocp/cmdha_24pt_anchorref_updatepos_convdepth_mult.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']