import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# {'General trash': 3966,
#  'Paper': 6352,
#  'Paper pack': 897,
#  'Metal': 936,
#  'Glass': 982,
#  'Plastic': 2943,
#  'Styrofoam': 1263,
#  'Plastic bag': 5178,
#  'Battery': 159,
#  'Clothing': 468}

"""
아래 코드를 모델 폴더 내 .py 파일에 넣고 loss_cls 수정
"""

num_ins = [3966, 6352, 897, 936, 982, 2943, 1263, 5178, 159, 468] # 각 클래스 별 데이터 개수
cnl_weights = [1 - (x/(sum(num_ins))) for x in num_ins] # weight 계산
class_weights = torch.FloatTensor(cnl_weights).to(device) # to Tensor




""" 사용 예시 """

num_ins = [3966, 6352, 897, 936, 982, 2943, 1263, 5178, 159, 468]
cnl_weights = [1 - (x/(sum(num_ins))) for x in num_ins]
class_weights = torch.FloatTensor(cnl_weights).to(device)

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     num_outs=5),
    # rpn_head=dict(
    #     type='RPNHead',
    #     in_channels=256,
    #     feat_channels=256,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         scales=[8],
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[4, 8, 16, 32, 64]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0,
            weight=class_weights # weight=class_weights 집어넣기
            ),
    #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # roi_head=dict(
    #     type='StandardRoIHead',
    #     bbox_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    #         out_channels=256,
    #         featmap_strides=[4, 8, 16, 32]),
    #     bbox_head=dict(
    #         type='Shared2FCBBoxHead',
    #         in_channels=256,
    #         fc_out_channels=1024,
    #         roi_feat_size=7,
    #         num_classes=80,
    #         bbox_coder=dict(
    #             type='DeltaXYWHBBoxCoder',
    #             target_means=[0., 0., 0., 0.],
    #             target_stds=[0.1, 0.1, 0.2, 0.2]),
    #         reg_class_agnostic=False,
    #         loss_cls=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0))
        ), 
    )

