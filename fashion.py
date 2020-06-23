{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "loading network...\n",
      "LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)\n",
      "classifying image...\n",
      "[[1.5952580e-11 5.1539533e-09 1.0000000e+00 2.6021700e-09]]\n",
      "shirt\n"
     ]
    }
   ],
   "source": [
    "from keras.preprocessing.image import img_to_array\n",
    "from keras.models import load_model\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import imutils\n",
    "import pickle\n",
    "import cv2\n",
    "\n",
    "image = cv2.imread('eb7a147e5b2c6b5b7d76c14ae7be4918.jpg')\n",
    "output = imutils.resize(image, width=400)\n",
    "image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "image = cv2.resize(image, (96, 96))\n",
    "image = image.astype(\"float\") / 255.0\n",
    "image = img_to_array(image)\n",
    "image = np.expand_dims(image, axis=0)\n",
    "\n",
    "\n",
    "print(\"loading network...\")\n",
    "model = load_model('E:/multi-label.h5', custom_objects={\"tf\": tf})\n",
    "categoryLB = pickle.loads(open('category_lb.pickle', \"rb\").read())\n",
    "colorLB = pickle.loads(open('color_lb.pickle', \"rb\").read())\n",
    "print(categoryLB)\n",
    "\n",
    "print(\"classifying image...\")\n",
    "(categoryProba, colorProba) = model.predict(image)\n",
    "print(categoryProba)\n",
    "categoryIdx = categoryProba[0].argmax()\n",
    "colorIdx = colorProba[0].argmax()\n",
    "categoryLabel = categoryLB.classes_[categoryIdx]\n",
    "colorLabel = colorLB.classes_[colorIdx]\n",
    "\n",
    "dict1 = {'category': categoryLabel, 'color': colorLabel }\n",
    "print(dict1['category'])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}