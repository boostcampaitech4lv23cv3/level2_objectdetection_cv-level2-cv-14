optimizer = dict(type="Adam", lr=0.0001, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=1e-05,
)
runner = dict(type="EpochBasedRunner", max_epochs=30)
