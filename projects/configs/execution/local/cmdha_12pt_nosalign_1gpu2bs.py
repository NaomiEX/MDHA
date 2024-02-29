_base_ = [
    "../../runtime_settings/1gpu2bs_run.py",
    "../../settings/cmdha_12pt_nosalign.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['MDHA', 'CustomDeformAttn', 'Projections', 'IQTransformerEncoder']