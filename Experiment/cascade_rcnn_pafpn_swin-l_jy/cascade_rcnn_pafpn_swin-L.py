_base_ = [
    # "./datasets/coco_detection.py",
    "./datasets/coco_detection_mix.py",
    "./models/cascade_rcnn_r50_pafpn-L.py",
    "./schedules/schedule_L.py",
    "./default_runtime.py",
]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"  # noqa
model = dict(
    type="CascadeRCNN",
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]),
)
