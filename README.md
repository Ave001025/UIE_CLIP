UIE_CLIP

train:CUDA_VISIBLE_DEVICES=2 python uie_main_train.py --opt_path options/NU2Net.yaml

test:python uie_main_test.py --opt_path options/NU2Net.yaml --test_ckpt_path "./uie_output/best_checkpoint.pth" --save_image
