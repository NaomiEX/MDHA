_base_ = [
    "../runtime_settings/1gpu2bs_run.py",
    "../settings/mdha_12pt_clean.py",
    "../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CircularDeformAttn', 'Projections', 'IQTransformerEncoder']