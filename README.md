# Unveiling the underwaterworld:CLIPperceptionmodel-guidedunderwaterimageenhancement

This repository contains the official implementation of the following paper:
> **Underwater Ranker: Learn Which Is Better and How to Be Better**<br>
> Chunle Guo<sup>#</sup>, Ruiqi Wu<sup>#</sup>, Xin Jin, Linghao Han, Zhi Chai, Weidong Zhang, Chongyi Li<sup>*</sup><br>
> Proceedings of the AAAI conference on artificial intelligence (AAAI), 2023<br>


### Training & Evaluation

Run the following commands for training:

```bash
python ranker_main_train.py --opt_path options/URanker.yaml
python uie_main_train.py --opt_path options.NU2Net.yaml
```

Run the following commands for evaluation:
```bash
python ranker_main_test.py --opt_path options/URanker.yaml --test_ckpt_path checkpoints/URanker_ckpt.pth
python uie_main_test.py --opt_path options.NU2Net.yaml --test_ckpt_path checkpoints/NU2Net_ckpt.pth --save_image
```

## Citation
If you find our repo useful for your research, please cite us:
```
@inproceedings{guo2023uranker,
  title={Underwater Ranker: Learn Which Is Better and How to Be Better},
  author={Guo, Chunle and Wu, Ruiqi and Jin, Xin and Han, Linghao and Chai, Zhi and Zhang, Weidong and Li, Chongyi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```


