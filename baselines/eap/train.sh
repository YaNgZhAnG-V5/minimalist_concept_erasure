echo "Train Text Attention Neg 3"
python train_flux_eap.py --prompt 'nudity' --train_method 'textattn' --devices '0,1' --negative_guidance 3 --gumbel_k_closest 200 --gumbel_num_centers 20 --gumbel_temp 2 --gumbel_hard 1 --gumbel_lr 1e-2 --lr 1e-4
