
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import imutils
import pickle
import cv2


def predict_clothes(file):
    image = cv2.imdecode(np.fromstring(
        file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    output = imutils.resize(image, width=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # loading network
    model = load_model('multi-label.h5', custom_objects={"tf": tf})
    categoryLB = pickle.loads(open('category_lb.pickle', "rb").read())
    colorLB = pickle.loads(open('color_lb.pickle', "rb").read())

    # classify image
    (categoryProba, colorProba) = model.predict(image)

    categoryIdx = categoryProba[0].argmax()
    colorIdx = colorProba[0].argmax()
    categoryLabel = categoryLB.classes_[categoryIdx]
    colorLabel = colorLB.classes_[colorIdx]

    clothes = {'category': categoryLabel, 'color': colorLabel}
    return clothes
