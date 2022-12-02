# dataset settings
dataset_type = "CocoDataset"
data_root = "../dataset/"
img_norm_cfg = dict(
    mean=[120.42, 114.62, 108.29], std=[56.894, 55.588, 57.481], to_rgb=True
)
albu_train_transforms = [
    dict(type="HorizontalFlip", p=0.5),
    dict(type="VerticalFlip", p=0.5),
    dict(type="RandomRotate90", p=0.5),
    dict(type="GaussNoise", p=0.5),
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5,
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="RGBShift",
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0,
            ),
            dict(
                type="HueSaturationValue",
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0,
            ),
        ],
        p=0.1,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=1.0),
            dict(type="MedianBlur", blur_limit=3, p=1.0),
        ],
        p=0.1,
    ),
]

train_pipeline = [
    dict(type="Mosaic", prob=0.5),
    dict(type="MixUp", prob=0.5),
    dict(type="Resize", img_scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_maske": "masks", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=[(512, 512), (640, 640), (768, 768), (896, 896), (1024, 1024)],
        flip=True,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "train.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_randomsplit_2022.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")
