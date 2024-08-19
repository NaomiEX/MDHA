# Local Execution Settings

The settings in this folder are only for testing on a local device with only 1-2 weak gpus. *It is not recommended to train the model with such a small batch size.* Please see the hpc/ execution settings otherwise, you can modify the local execution settings by simply changing the loaded runtime_settings/ to one that uses a bigger batch size. For ex. changing "../../runtime_settings/2gpu2bs_run.py" -> "../../runtime_settings/4gpu16bs_run.py". See runtime_settings/ for all pre-defined gpu-batchsize combinations.