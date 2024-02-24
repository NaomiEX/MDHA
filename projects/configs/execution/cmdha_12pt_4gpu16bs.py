_base_ = [
    "../runtime_settings/4gpu16bs_run.py",
    "../settings/cmdha_12pt_nocp.py",
    "../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'IQTransformerEncoder']