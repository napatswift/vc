from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from label_studio_tools.core.utils.params import get_env
from label_studio_tools.core.utils.io import get_local_path

import cv2
from gradio_client import Client
import json

model_client = Client("https://napatswift-votecount-ml-be.hf.space/")

class MyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

        self.from_name, schema = list(self.parsed_label_config.items())[0]
        self.model_version = '0.0.1'
        self.schema = {
            'to_name': schema['to_name'][0],
            'from_name': self.from_name,
            'type': schema['type'],
            'value_key': schema['inputs'][0]['value']
        }

        self.model_api = model_client

    def predict(self, tasks, **kwargs):
        prediction = []

        image_value_key = self.schema['value_key']
        for task in tasks:
            image_path = task['data'][image_value_key]
            print('Task ID', task['id'], 'image path', image_path)
            if image_path.startswith('/data/local-files/?d='):
                image_path = image_path.replace('/data/local-files/?d=','/')
            
            try:
                results, score = self._predict(image_path)
            except Exception as e:
                results = []
                score = 0.0

            image_height, image_width = cv2.imread(image_path).shape[:2]

            prediction.append({
                'result': [
                    {
                        'from_name': self.schema['from_name'],
                        'to_name': self.schema['to_name'],
                        'type': 'rectanglelabels',
                        'value': self._get_x_y_w_h(result, image_width, image_height)
                    } for result in results
                ],
                'score': score,
                'model_version': self.model_version
            })

        return prediction

    def _predict(self, image_path):
        """
        Predicts the bounding boxes of objects in an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            list: A list of bounding boxes, in the format `[x1, y1, x2, y2]`.
            float: The score of the prediction.

        Raises:
            Exception: If an error occurs during prediction.
        """



        # Get the result file path.
        print('Predicting image', image_path)
        result_file_path = self.model_api.predict(
            image_path, api_name="/predict")

        # Load the result file.
        with open(result_file_path) as fp:
            loaded_data = json.load(fp)

        # Get the first prediction.
        prediction = loaded_data['predictions'][0]

        # Get the bounding boxes and score.
        bounding_boxes, scores = self._prediction_filter(prediction['det_polygons'],
                                                         prediction['det_scores'])

        score = self._calc_score(scores)

        # Return the bounding boxes, and score.
        return bounding_boxes, score

    def _prediction_filter(self, polygons, scores, score_threshold=0.5):
        filtered_polygons = []
        filtered_scores = []
        for polygon, score in zip(polygons, scores):
            if score >= score_threshold:
                filtered_polygons.append(polygon)
                filtered_scores.append(score)
        return filtered_polygons, filtered_scores

    def _calc_score(self, scores):
        return sum(scores) / len(scores)

    def _get_x_y_w_h(self, polygon: list, image_width, image_height):
        """
        Get the x, y, w, and h coordinates of a rectangle from a polygon.

        Args:
            polygon (list): A list of 2D points that define the polygon.
            image_width (int): The width of the image in pixels.
            image_height (int): The height of the image in pixels.

        Returns:
            dict: A dictionary with the following keys:
                * x: The x-coordinate of the top-left corner of the rectangle.
                * y: The y-coordinate of the top-left corner of the rectangle.
                * width: The width of the rectangle in pixels.
                * height: The height of the rectangle in pixels.
                * rectanglelabels: A list of labels for the rectangle.
        """

        # Get the x-coordinates of the polygon.
        x_coordinates = [pos/image_width*100 for i,
                         pos in enumerate(polygon) if i % 2 == 0]

        # Get the y-coordinates of the polygon.
        y_coordinates = [pos/image_height*100 for i,
                         pos in enumerate(polygon) if i % 2 != 0]

        # Get the minimum and maximum x-coordinates.
        min_x = min(x_coordinates)
        max_x = max(x_coordinates)

        # Get the minimum and maximum y-coordinates.
        min_y = min(y_coordinates)
        max_y = max(y_coordinates)

        # Calculate the width and height of the rectangle.
        width = max_x - min_x
        height = max_y - min_y

        # Create a dictionary with the x, y, width, and height of the rectangle.
        rectangle = {
            "x": min_x,
            "y": min_y,
            "width": width,
            "height": height,
            "rectanglelabels": ["Text"]
        }

        # Return the rectangle.
        return rectangle
