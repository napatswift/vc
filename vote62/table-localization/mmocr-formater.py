import json
import os
import subprocess
import argparse
from sklearn.model_selection import train_test_split

def export(split_set, split_name):
    dataset = {'images': list(), 'annotations': list(), 'categories': annot['categories']}
    for image in split_set:
        for annotation in annot['annotations']:
            if annotation['image_id'] == image['id']:
                dataset['annotations'].append(annotation)
        dataset['images'].append(image)
    with open(os.path.join(dir_path, f'{split_name}_coco.json'), 'w') as fp:
        json.dump(dataset, fp, indent=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', 'path-to-coco-directory')
    args = parser.parse_args()
    dir_path = args.path
    img_dir = 'images'

    with open(os.path.join(dir_path, 'result.json')) as fp:
        annot = json.load(fp)

    for img in annot['images']:
        if subprocess.call(
            ['cp', img['file_name'], os.path.join(dir_path, img_dir)]
        ) != 0: # if success
            print(img)
        else:
            img['file_name'] = os.path.join(img_dir, os.path.basename(img['file_name']))
    X = []
    Y = []
    for img in annot['images']:
        img_id = img['id']
        y_list = []
        for y in annot['annotations']:
            if y['image_id'] == img_id:
                y_list.append(y)
        Y.append(len(y_list))
        X.append(img)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.15)

    export(x_train, 'train')
    export(x_test, 'test')
