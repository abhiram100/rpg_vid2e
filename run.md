# First Upsample
LD_LIBRARY_PATH=/data/storage/abhiram/miniforge3/envs/v2e/lib/python3.10/site-packages/nvidia/cudnn/lib/

```bash
CUDA_VISIBLE_DEVICES=3 python upsampling/upsample.py --input_dir=N-Caltech101/original/airplanes/ --output_dir=N-Caltech101/upsampled/airplanes/ 
```

# Generate Events
```bash
python esim_torch/scripts/generate_events.py --input_dir=/data/storage/abhiram/rpg_vid2e_N-Caltech101/ncaltech_imgs_vid2e/background_google/image_0031 \
                                     --output_dir=N-Caltech101/events_test/ \
                                     --contrast_threshold_neg=0.2 \
                                     --contrast_threshold_pos=0.2 \
                                     --refractory_period_ns=0
```
