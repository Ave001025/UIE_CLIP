# Unveiling the underwater world:CLIP perception model-guided underwater image enhancement

This repository contains the official implementation of the following paper:
> **Unveiling the underwater world:CLIP perception model-guided underwater image enhancement**<br>
> Jiangzhong Cao<sup>#</sup>,Zekai Zeng<sup>#</sup>, Xu Zhang, Huan Zhang, Chunling Fan, Gangyi Jiang,  Weisi Lin<sup>*</sup><br>
> *Pattern Recognition*, 2025<br>


### Training & Evaluation

Run the following commands for training:

```bash
CUDA_VISIBLE_DEVICES=2 python uie_main_train.py --opt_path options/NU2Net.yaml
```

Run the following commands for evaluation:
```bash
python uie_main_test.py --opt_path options/NU2Net.yaml --test_ckpt_path "./uie_output/best_checkpoint.pth" --save_image
```

## Citation
If you find our repo useful for your research, please cite us:
```
@inproceedings{guo2023uranker,
  title={Unveiling the underwater world:CLIP perception model-guided underwater image enhancement},
  author={Jiangzhong Cao,Zekai Zeng,Xu Zhang,Huan Zhang,Chunling Fan,Gangyi Jiang,Weisi Lin},
  booktitle={Pattern Recognition},
  year={2025}
}
```


