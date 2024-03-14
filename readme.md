# MDHA

## setup environment

### 1. create env:
```bash
conda create --name [env_name] python=3.8
conda activate [env_name]
```

### 2. install requirements:

```bash
pip install -r requirementsv3.txt
```

### 3. create symbolic link to data

```bash
ln -s /path/to/nuscenes ./data/nuscenes
```

(local):
```bash
mkdir ./data
ln -s /home/junn/Michelle/data/nuscenes ./data/nuscenes
```

(slurm)
```bash
ln -s /ibm/gpfs/home/made0008/data/nuscenes ./data/nuscenes
```

### 4. compile ops

#### 4.1. daf ops
(local)
```bash
cd projects/mmdet3d_plugin/attentions/daf_ops
python3 setup.py develop
cd ../../../../
```

#### 4.2. deformable ops
```bash
cd projects/mmdet3d_plugin/attentions/deform_ops
./make.sh
cd ../../../../
```

### 5. create annotation infos
```bash
python3 tools/nuscenes_converter.py --trainval-path data/nuscenes --test-path data/nuscenes/test --info_out_path data/nuscenes/anno --version v1.0-trainval,v1.0-test
```


## create info annotations
(local)
```bash
python3 tools/nuscenes_converter.py --version v1.0-trainval,v1.0-test --trainval-path data/nuscenes --test-path data/nuscenes/test --info_out_path ./data/nuscenes/anno
```

(slurm)

note: check that the conda env is correct and discard any caches in the ops folders
```bash
sbatch compile.job
```

## run
(local):
```bash
tools/dist_train.sh projects/configs/execution/local/cmdha_4ptenc_24ptdec_anchorref_convdepth_mult_updatepos_newproj_bind_1gpu2bs_25ep.py 1 --work-dir work_dirs/cmdha_4ptenc_24ptdec_anchorref_convdepth_mult_updatepos_newproj_bind_1gpu2bs_25ep
```

(local, 2gpu):
```bash
tools/dist_train.sh projects/configs/execution_settings/s4dv3_2gpu_8bs.py 2 --work-dir work_dirs/test
```

(slurm):
NOTE: for bs48, MEM=100-120G should work, for bs32, MEM=80G works
NOTE: use either gpu2,3,6
```bash
GPUS_PER_NODE=2 MEM=160G ./tools/slurm_train.sh highprio mdha_12 2 ./projects/configs/execution/cmdha_12pt_nopos3d_4gpu16bs.py ./work_dirs/cmdha_12pt_nopos3d_4gpu16bs
```

# 4 GPU
GPUS_PER_NODE=4 MEM=160G ./tools/slurm_train.sh highprio mdha_4_12_wrap_abl 4 ./projects/configs/execution/hpc/cmdha_4ptenc_12ptdec_anchorref_convdepth_mult_updatepos_wrap_4gpu16bs_25ep.py ./work_dirs/cmdha_4ptenc_12ptdec_anchorref_convdepth_mult_updatepos_wrap_4gpu16bs_25ep_stable

# 8 GPU
NODES=2 CPUS_PER_TASK=12 GPUS_PER_NODE=4 MEM=160G ./tools/slurm_train.sh highprio mdha_r101 8 ./projects/configs/execution/hpc/cmdha_r101_4ptenc_12ptdec_anchorref_updatepos_v3_8gpu16bs.py ./work_dirs/padding

## eval
(local):
```bash
tools/dist_test.sh ./projects/configs/execution_settings/cmdha_12pt_anchorref_convdepth_mult_updatepos_1gpu2bs.py work_dirs/from_hpc/cmdha_12pt_anchorref_convdepth_mult_updatepos_4gpu16bs/iter_165252.pth 1 --eval bbox
```

(hpc)
GPUS=4 GPUS_PER_NODE=4 CPUS_PER_TASK=12 MEM=160G tools/slurm_test.sh highprio r101_eval ./projects/configs/execution/hpc/cmdha_r101_4ptenc_24ptdec_anchorref_updatepos_v2_8gpu16bs.py ./work_dirs/cmdha_r101_4ptenc_24ptdec_anchorref_updatepos_v2_8gpu16bs_resume/latest.pth

(test all):
(2 GPUS)
./test_multiple.sh ./projects/configs/execution/local/cmdha_4ptenc_24ptdec_anchorref_convdepth_mult_updatepos_2gpu2bs.py work_dirs/from_hpc/cmdha_4ptenc_24ptdec_anchorref_convdepth_mult_updatepos_4gpu16bs_25ep/ 2

## benchmark
python tools/benchmark.py ./projects/configs/execution/local/cmdha_4ptenc_24ptdec_anchorref_convdepth_mult_updatepos_newproj_2gpu2bs.py --samples 300