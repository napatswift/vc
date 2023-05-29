import json
import os
import os.path as osp
import argparse


def convert_bbox(bbox):
    x0, y0, x1, y1 = bbox
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def coco_annotation_to_mmocr_textdet_annotation(coco_annotation):
    return dict(
        polygon=convert_bbox(coco_annotation['bbox']),
        bbox=coco_annotation['bbox'],
        bbox_label=coco_annotation['category_id'],
        ignore=bool(coco_annotation['ignore']),
    )


def get_file_name(file_path):
    return os.path.basename(file_path)


def converter(coco_json):
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    # Create dict for annotation using image_id as key
    annotations = coco['annotations']
    annotations_dict = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        annotations_dict[image_id].append(
            coco_annotation_to_mmocr_textdet_annotation(annotation)
        )

    categories = coco['categories']
    data_list = []
    for image in coco['images']:
        annotation = dict(
            img_path=get_file_name(image['file_name']),
            height=image['height'],
            width=image['width'],
            instances=annotations_dict[image['id']],
        )
        data_list.append(annotation)

    return dict(
        metainfo=dict(
            dataset_type='TextDetDataset',
            task_name='textdet',
            category=categories
        ),
        data_list=data_list,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_json', type=str,)
    args = parser.parse_args()

    output_dir = osp.dirname(args.coco_json)

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # get coco json file name
    file_name = get_file_name(args.coco_json).split('.')[0]

    with open(osp.join(output_dir, f'mmocr-dataset-{file_name}.json'), 'w') as f:
        json.dump(converter(args.coco_json), f, indent=1)
