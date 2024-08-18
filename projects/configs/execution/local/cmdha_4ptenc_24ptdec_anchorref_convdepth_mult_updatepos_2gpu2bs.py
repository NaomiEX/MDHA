_base_ = [
    "../../runtime_settings/2gpu2bs_run.py",
    "../../settings/nocp/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult_newproj_nocp.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'AnchorEncoder',
               'AnchorRefinement', 'ReferencePoints']