optimizer = dict(type="Adam", lr=0.0001, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=50, warmup_ratio=0.0001, step=[4, 11]
)
runner = dict(type="EpochBasedRunner", max_epochs=12)
