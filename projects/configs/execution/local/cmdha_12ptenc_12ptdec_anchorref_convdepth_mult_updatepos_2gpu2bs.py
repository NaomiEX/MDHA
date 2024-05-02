_base_ = [
    "../../runtime_settings/2gpu2bs_run.py",
    "../../settings/nocp/cmdha_12ptenc_12ptdec_anchorref_updatepos_convdepth_mult_newproj_nocp.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints']