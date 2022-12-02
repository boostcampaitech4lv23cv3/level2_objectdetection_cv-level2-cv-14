# optimizer
optimizer = dict(type="Adam", lr=0.00001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=2000, warmup_ratio=0.1, step=[218, 246]
)
runner = dict(type="EpochBasedRunner", max_epochs=45)
