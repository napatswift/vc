import tensorflow as tf
import pandas as pd
import os

def classify_images(csv_path, model_path, batch_size=32):
    # Load CSV file containing image paths
    df = pd.read_csv(csv_path, sep='\t')
    pred_file_path = os.path.join(os.path.dirname(csv_path),'pred_classes.txt')
    predicted_images = []
    if os.path.exists(pred_file_path):
        with open(pred_file_path, 'r') as fp:
            predicted_images = [line.split()[0] for line in fp.read().split('\n') if line]
        pred_file = open(pred_file_path, 'a')
    else:
        pred_file = open(pred_file_path, 'w')
    
    # Load pre-trained TensorFlow model
    model = tf.keras.models.load_model(model_path)
    
    image_paths = df[~df['image'].isin(predicted_images)]['image']
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for image_path in batch_paths:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_arr = tf.keras.preprocessing.image.img_to_array(img)
            img_arr = tf.keras.applications.mobilenet_v3.preprocess_input(img_arr)
            batch_images.append(img_arr)

        preds = model.predict(tf.convert_to_tensor(batch_images))
        cls_ids = preds.argmax(1)
        for img, p in zip(batch_paths, preds):
            pred_file.write(img)
            pred_file.write(' ')
            pred_file.write(str(p.argmax()))
            pred_file.write(' ')
            pred_file.write(str(p.max()))
            pred_file.write('\n')
    
    pred_file.close()


if __name__ == '__main__':
    classify_images('data/metadata.txt', 'saved_models/mobile-net')