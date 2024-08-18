_base_ = [
    "../../runtime_settings/1gpu2bs_run.py",
    "../../settings/nocp/cmdha_4ptenc_12ptdec_anchorref_updatepos_convdepth_mult_newproj_multref_wrap_nocp.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'AnchorEncoder',
               'AnchorRefinement', 'ReferencePoints']