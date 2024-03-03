_base_ = [
    "../../runtime_settings/2gpu6bs_25ep_run.py",
    "../../settings/ablation/cmdha_12ptdecoder_noenc_full.py",
    "../../runtime_settings/debug.py"
]

debug_modules=['MDHA',  'CustomDeformAttn', 'Projections', 'IQTransformerEncoder',
               'PETRTransformerDecoder', 'StreamPETR']