_base_ = "base.py"

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        sup=dict(
            ann_file="/opt/ml/dataset/train_randomsplit_2022.json",
            img_prefix="/opt/ml/dataset/train/",
        ),
        unsup=dict(
            ann_file="/opt/ml/dataset/test.json",
            img_prefix="/opt/ml/dataset/test/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)
