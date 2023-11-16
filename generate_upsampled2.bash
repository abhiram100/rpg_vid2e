for class in "anchor" "ant" "background_google" 
do
    file=/data/storage/abhiram/rpg_vid2e_N-Caltech101/ncaltech_imgs_vid2e/$class
    if [ -d "N-Caltech101/events_upsample/$class" ]; then
        # Take action if $DIR exists. 
        echo "Events exist, skipping!"
        echo $file
        echo ${file:(65+${#class}+1):10}
        echo ""
    else
        echo "Generating events..."
        echo $file
        echo ""
        CUDA_VISIBLE_DEVICES=4 python upsampling/upsample.py --input_dir=$file \
                                         --output_dir=N-Caltech101/upsampled/$class

        python esim_torch/scripts/generate_events.py --input_dir=N-Caltech101/upsampled/$class \
                                            --output_dir=N-Caltech101/upsampled_events/$class \
                                            --contrast_threshold_neg=0.2 \
                                            --contrast_threshold_pos=0.2 \
                                            --refractory_period_ns=0
    fi
done
 
