from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
import cv2
from gradio_client import Client
import json


class MyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(MyModel, self).__init__(**kwargs)
        # print('parsed_label_config', self.parsed_label_config)
        self.from_name, schema = list(self.parsed_label_config.items())[0]
        self.model_version = '0.0.1'
        self.schema = {
            'to_name': schema['to_name'][0],
            'from_name': self.from_name,
            'type': schema['type'],
            'value_key': schema['inputs'][0]['value']
        }

        self.model_api = Client("https://napatswift-test.hf.space/")

    def predict(self, tasks, **kwargs):
        prediction = []

        # ...and implement predict function
        # print('Predicting tasks:', tasks)
        image_value_key = self.schema['value_key']
        for task in tasks:
            image_path = get_image_local_path(task['data'][image_value_key])

            try:
                results = self._predict(image_path)
            except Exception as e:
                results = []

            image_height, image_width = cv2.imread(image_path).shape[:2]
            pred_bboxes = [self._get_x_y_w_h(
                result, image_width, image_height) for result in results]
            pred_bboxes = [
                bbox for bbox in pred_bboxes if self._bbox_filter(bbox)]
            prediction.append({
                'result': [
                    {
                        'from_name': self.schema['from_name'],
                        'to_name': self.schema['to_name'],
                        'type': 'rectanglelabels',
                        'value': self._get_x_y_w_h(result, image_width, image_height)
                    } for result in results
                ],
                'score': 1.0,
                'model_version': self.model_version
            })

        print('prediction', prediction)
        return prediction

    def _bbox_filter(self, bbox):
        """Filter bbox with width and height > 0 and x, y > 0 and x, y, x+with, y+height < 100"""
        return (bbox['width'] > 0 and
                bbox['height'] > 0 and
                bbox['x'] > 0 and
                bbox['y'] > 0 and
                bbox['x'] + bbox['width'] < 100 and
                bbox['y'] + bbox['height'] < 100)

    def _predict(self, image_path):
        result = self.model_api.predict(image_path, api_name="/predict")
        with open(result) as fp:
            return json.load(fp)['predictions'][0]['det_polygons']

    def _get_x_y_w_h(self, polygon: list, image_width, image_height):
        """Get x, y, w, h from polygon

        Args:
            polygon (list): [x0, y0, x1, y1, ..., xn, yn]
        """
        x_pos = [pos/image_width*100 for i,
                 pos in enumerate(polygon) if i % 2 == 0]
        y_pos = [pos/image_height*100 for i,
                 pos in enumerate(polygon) if i % 2 != 0]
        x = min(x_pos)
        y = min(y_pos)
        w = max(x_pos) - x
        h = max(y_pos) - y
        return dict(x=x,
                    y=y,
                    width=w,
                    height=h,
                    rectanglelabels=['Text'])
