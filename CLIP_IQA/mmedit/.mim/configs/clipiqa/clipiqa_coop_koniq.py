exp_name = 'clipiqa_coop_koniq_224_Clear_Turbid_random_best'

# model settings
model = dict(
    type='CLIPIQA',
    generator=dict(
        type='CLIPIQAPredictor',
        backbone_name='RN50',
        classnames=[
            #['Good photo.', 'Bad photo.'],
            #['Clear underwater photo.','Unclear underwater photo.'],
            #['Clean underwater photo.','Noisy underwater photo.'],
            ['Clear underwater photo.','Turbid underwater photo.']
            
        ]),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['L1DIS'], crop_border=0)

# dataset settings
train_dataset_type = 'IQAKoniqDataset'
val_dataset_type = 'IQAKoniqDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        to_rgb=True),
    #dict(type='Resize', keys=['lq'], scale=1/2, keep_ratio=True),
    dict(type='Resize', keys=['lq'], scale=(224, 224)),
    dict(
        type='Flip', keys=['lq'], flip_ratio=0.5,
        direction='horizontal'),
    # dict(type='Flip', keys=['lq'], flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomTransposeHW', keys=['lq'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='ToTensor', keys=['gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        backend='pillow'),
    # dict(type='Resize_PIL', keys=['lq'], scale=512, interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        to_rgb=True),
    # dict(type='Resize', keys=['lq'], scale=1/4, keep_ratio=True),
    #dict(type='Resize', keys=['lq'], scale=(512, 512)),
    dict(type='Resize', keys=['lq'], scale=(224, 224)),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='ToTensor', keys=['gt'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Resize', keys=['lq'], scale=(224, 224)),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=64, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    #有点不太懂，这样子训练的话，传输进去的数据格式？
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=train_dataset_type,
            #图像文件（应该改一下下面这两段代码就可以了）
            #img_folder="/data/zengzekai/UIE/Datasets/koniq10k/1024x768/",
            #相应的配置文件，包括分数，数据集划分等
            #ann_file="/data/zengzekai/UIE/Datasets/koniq10k/koniq10k_distributions_sets.csv",
            
            img_folder="/data/zengzekai/UIE/Datasets/SOTA_Dataset",
            ann_file="/data/zengzekai/UIE/Datasets/SOTA_dataset.mat",
            
            pipeline=train_pipeline,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        #img_folder="/data/zengzekai/UIE/Datasets/koniq10k/1024x768/",
        #ann_file="/data/zengzekai/UIE/Datasets/koniq10k/koniq10k_distributions_sets.csv",
        img_folder="/data/zengzekai/UIE/Datasets/SOTA_Dataset",
        ann_file="/data/zengzekai/UIE/Datasets/SOTA_dataset.mat",
        pipeline=test_pipeline,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        #img_folder="/data/zengzekai/UIE/Datasets/koniq10k/1024x768/",
        #ann_file="/data/zengzekai/UIE/Datasets/koniq10k/koniq10k_distributions_sets.csv",
        img_folder="/data/zengzekai/UIE/Datasets/SOTA_Dataset",
        ann_file="/data/zengzekai/UIE/Datasets/SOTA_dataset.mat",
        pipeline=test_pipeline,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='SGD',
        lr=0.002))

# learning policy
total_iters = 100000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[1000000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
#evaluation = dict(interval=100, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
