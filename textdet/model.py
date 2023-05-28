from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
import logging
import cv2
from gradio_client import Client
import json

model_client = Client("https://napatswift-votecount-ml-be.hf.space/")


class MyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

        # Get the name of the field in the input data that contains the text to be classified.
        self.from_name, schema = list(self.parsed_label_config.items())[0]

        # Set the model version.
        self.model_version = '0.1.0'

        # Set the schema of the input data.
        self.schema = {
            'to_name': schema['to_name'][0],
            'from_name': self.from_name,
            'type': schema['type'],
            'value_key': schema['inputs'][0]['value']
        }

        # Create the API client for the model.
        self.model_api = model_client

    def predict(self, tasks, **kwargs):
        """
        Predicts the bounding boxes of objects in an image.

        Args:
            tasks: A list of tasks.
            kwargs: Additional keyword arguments.

        Returns:
            A list of predictions, each of which contains a list of bounding boxes, a score, and the model version.

        Raises:
            ValueError: If the schema does not contain a `from_name` or `to_name` key.
        """

        # Check the schema.
        if not self.schema.get('from_name'):
            raise ValueError('Schema must contain a `from_name` key.')
        if not self.schema.get('to_name'):
            raise ValueError('Schema must contain a `to_name` key.')
        
        image_value_key = self.schema['value_key']

        # Predict the bounding boxes for each task.
        prediction = []

        for task in tasks:
            image_path = task['data'][image_value_key]
            logging.info('Predicting image %s', image_path)
            print('Task ID', task['id'], 'image path', image_path)
            if image_path.startswith('/data/local-files/?d='):
                image_path = image_path.replace('/data/local-files/?d=', '/')
            elif image_path.startswith('/data/upload/'):
                image_path = get_image_local_path(image_path)
            
            logging.info('Predicting image %s', image_path)

            try:
                results, score = self._predict(image_path)
            except Exception as e:
                results = []
                score = 0.0

            # Get the image height and width.
            image_height, image_width = cv2.imread(image_path).shape[:2]

            # Create a prediction for each bounding box.
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

        # Return the predictions.
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

        # Calculate the score.
        score = self._calc_score(scores)

        # Return the bounding boxes, and score.
        return bounding_boxes, score

    def _prediction_filter(self, polygons, scores, score_threshold=0.5):
        """
        Filters the predictions based on the score threshold.
        
        Args:
            polygons (list): A list of polygons.
            scores (list): A list of scores.
            score_threshold (float): The score threshold. defaults to 0.5.
        
        Returns:
            list: A list of filtered polygons.
            list: A list of filtered scores.
        """

        filtered_polygons = []
        filtered_scores = []
        # Iterate over the polygons and scores.
        for polygon, score in zip(polygons, scores):

            # If the score is greater than or equal to the score threshold,
            # then add the polygon and score to the filtered lists.
            if score >= score_threshold:
                filtered_polygons.append(polygon)
                filtered_scores.append(score)
        return filtered_polygons, filtered_scores

    def _calc_score(self, scores):
        """
        Calculates the score of a prediction.

        Args:
            scores (list): A list of scores.

        Returns:
            float: The score of the prediction.

        Raises:
            ValueError: If the list of scores is empty.
        """

        if not scores:
            raise ValueError("The list of scores cannot be empty.")

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
