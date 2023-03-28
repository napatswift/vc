_base_ = 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Table',)
data = dict(
    train=dict(
        classes=classes,
        img_prefix='data/table-det-610/',
        ann_file='data/table-det-610/train_coco.json'),
    val=dict(
        classes=classes,
        ann_file='data/table-det-610/test_coco.json'),
    test=dict(
        classes=classes,
        ann_file='data/table-det-610/test_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
