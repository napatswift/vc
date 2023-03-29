_base_ = 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

classes = ('Table',)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes)),))

dataset_type = 'COCODataset'
data = dict(
    train=dict(
        classes=classes,
        img_prefix='data/table-det-610/',
        ann_file='data/table-det-610/train_coco.json'),
    val=dict(
        classes=classes,
        img_prefix='data/table-det-610/',
        ann_file='data/table-det-610/test_coco.json'),
    test=dict(
        classes=classes,
        img_prefix='data/table-det-610/',
        ann_file='data/table-det-610/test_coco.json'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Translate',),
    dict(type='Rotate',level=40,),
    dict(type='Shear',prob=0.6,),
    dict(type='ColorTransform',prob=0.7),
    dict(type='BrightnessTransform',prob=0.7),
    dict(type='Normalize', **dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
optimizer = dict(lr=1e-3)

load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
