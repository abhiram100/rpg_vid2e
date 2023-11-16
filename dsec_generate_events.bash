LD_LIBRARY_PATH=/data/storage/abhiram/miniforge3/envs/v2e/lib/python3.10/site-packages/nvidia/cudnn/lib/
# test split
for place in "interlaken_00_a" "interlaken_00_b" "interlaken_01_a" "thun_01_a" "thun_01_b" "thun_02_a" "zurich_city_12_a" "zurich_city_13_a" "zurich_city_13_b" "zurich_city_14_a" "zurich_city_14_b" "zurich_city_14_c" "zurich_city_15_a" 
do
    image_folder=/data/storage/abhiram/dsec/dsec-det/dsec_merged_root/test/$place/images/left/distorted
    output_dir=/data/storage/abhiram/dsec/dsec-det/vid2e_dsec_merged_root/test/$place
    echo $output_dir/upsampled
    
    # Generate upsampled images
    CUDA_VISIBLE_DEVICES=2 python upsampling/upsample.py \
                                  --input_dir=$image_folder \
                                  --output_dir=$output_dir/upsampled
    
    # Generate the events
    # CUDA_VISIBLE_DEVICES=0 python esim_torch/scripts/dsec_generate_events.py --input_dir=$output_dir/upsampled \
    #                                         --output_dir=$output_dir \
    #                                         --contrast_threshold_neg=0.2 \
    #                                         --contrast_threshold_pos=0.2 \
    #                                         --refractory_period_ns=0
    break
done

