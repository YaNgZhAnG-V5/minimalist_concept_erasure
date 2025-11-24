#! /bin/bash

exp=flux_inference # beta_ablation, flux_inference, plot_side_by_side, nudenet, esd_training

# beta ablation
if [ $exp == "beta_ablation" ]; then
    for beta in 0.1 0.01 0.001 0.0001 0.00001; do
        python scripts/train.py -c "people with gun" --cfg-options trainer.device=cuda:1 trainer.beta=${beta}  logger.project="jan23_people_with_gun"
    done
fi

# flux inference to generate original and unlearn images
if [ $exp == "flux_inference" ]; then
    for prompt in nudity_concept ring-a-bell-3-16 ring-a-bell-3-38 ring-a-bell-3-75 p4d i2p p4d mma; do
        python scripts/flux_inference.py --prompt ${prompt} --save_dir images/sld --device 3 --baseline sld -c nude
    done
fi

# plot flux unlearn vs original side by side
if [ $exp == "plot_side_by_side" ]; then
    for prompt in ring-a-bell-3-16 ring-a-bell-3-38 ring-a-bell-3-75 p4d; do
        python scripts/plot_imgs_side_by_side.py \
            --folder1 images/1024x1024/${prompt}/flux_unlearn \
            --folder2 images/1024x1024/${prompt}/original_image \
            --target-folder images/1024x1024/${prompt}/flux_unlearn_vs_original \
            --replace-folder True
    done
fi

# nudenet detection
if [ $exp == "nudenet" ]; then
    for prompt in ring-a-bell-3-16 ring-a-bell-3-38 ring-a-bell-3-75 p4d nudity_concept i2p p4d mma; do
        for type in flux_unlearn original_image; do
            echo "Running nudenet for ${prompt} ${type}"
            python scripts/run_nudenet.py -t ./images/nudity_flow_edit/${prompt}/${type}
        done
    done
fi

# esd training
if [ $exp == "esd_training" ]; then
    for prompt in i2p nudity; do
        for method in noattn selfattn_one selfattn_two textattn notime; do
            python baselines/esd/train.py --prompt ${prompt} --train_method ${method}
        done
    done
fi
