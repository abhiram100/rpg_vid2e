for class in "bonsai" "brain" "brontosaurus" "buddha" "butterfly" "camera" "cannon" "car_side" "ceiling_fan" "cellphone" "chair" "chandelier" "cougar_body" "cougar_face" "crab" "crayfish" "crocodile" "crocodile_head" "cup" "dalmatian" "dollar_bill" "dolphin" "dragonfly" "electric_guitar" "elephant" "emu" "euphonium" "ewer" "faces_easy" "ferry" "flamingo" "flamingo_head" "garfield" "gerenuk" "gramophone" "grand_piano" "hawksbill" "headphone" "hedgehog" "helicopter" "ibis" "inline_skate" "joshua_tree" "kangaroo" "ketch" "lamp" "laptop" "leopards" "llama" "lobster" "lotus" "mandolin" "mayfly" "menorah" "metronome" "minaret" "motorbikes" "nautilus" "octopus" "okapi" "pagoda" "panda" "pigeon" "pizza" "platypus" "pyramid" "revolver" "rhino" "rooster" "saxophone" "schooner" "scissors" "scorpion" "sea_horse" "snoopy" "soccer_ball" "stapler" "starfish" "stegosaurus" "stop_sign" "strawberry" "sunflower" "tick" "trilobite" "umbrella" "watch" "water_lilly" "wheelchair" "wild_cat" "windsor_chair" "wrench" "yin_yang" 
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
        CUDA_VISIBLE_DEVICES=1 python esim_torch/scripts/generate_events1.py --input_dir=$file \
                                            --output_dir=N-Caltech101/vid2e_new_thresh/$class \
                                            --contrast_threshold_neg=0.06 \
                                            --contrast_threshold_pos=0.06 \
                                            --refractory_period_ns=0
    fi
done