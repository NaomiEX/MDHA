_base_ = [
    "../../runtime_settings/2gpu2bs_run.py",
    "../../settings/ablation/cmdha_4ptenc_24ptdec_anchorref_updatepos_convdepth_mult_newproj_bind.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'AnchorRefinement', 'ReferencePoints', 'DepthNet']