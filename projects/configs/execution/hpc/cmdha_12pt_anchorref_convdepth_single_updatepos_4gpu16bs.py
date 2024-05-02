_base_ = [
    "../../runtime_settings/4gpu16bs_run.py",
    "../../settings/nocp/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_single.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']