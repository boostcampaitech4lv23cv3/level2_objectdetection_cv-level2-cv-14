ex_base = "/opt/ml/baseline/Experiment/cascade_rcnn_pafpn_convnext-t"
_base_ = [
    f"{ex_base}/datasets/coco_detection.py",
    f"{ex_base}/models/cascade_rcnn_r50_pafpn.py",
    f"{ex_base}/schedules/schedule_adamw_1x.py",
    f"{ex_base}/default_runtime.py",
]

custom_imports = dict(imports=["mmcls.models"], allow_failed_imports=False)
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth"  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type="mmcls.ConvNeXt",
        arch="tiny",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained", checkpoint=checkpoint_file, prefix="backbone."
        ),
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
)
