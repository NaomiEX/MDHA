_base_ = [
    "../../runtime_settings/4gpu16bs_run_25ep.py",
    "../../settings/nocp/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult_wrap_newproj_nocp.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'AnchorEncoder',
               'AnchorRefinement', 'ReferencePoints']