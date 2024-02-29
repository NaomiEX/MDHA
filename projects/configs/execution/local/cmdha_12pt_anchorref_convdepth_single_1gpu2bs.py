_base_ = [
    "../runtime_settings/1gpu2bs_run.py",
    "../settings/cmdha_12pt_anchorref_convdepth_single.py",
    "../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']