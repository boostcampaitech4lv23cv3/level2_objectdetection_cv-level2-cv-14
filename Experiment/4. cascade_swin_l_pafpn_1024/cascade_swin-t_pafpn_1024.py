_base_ = [
    './datasets/coco_detection.py',
    './models/cascade_swin_l_pafpn_1024.py',
    './schedules/schedule_adamw_1x.py',
    './default_runtime.py',
]

