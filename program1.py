#Importing all the important models and install them if not installed on your device
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Setting an HTTPS Context to fetch data from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

#Fetching the data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['A','a','B','b','C','c','D','d','E','e','F','f','G','g','H','h','I','i','J','j','K','k','L','l','M','m','N','O','o','P','p','Q','q','R','r','S','s','T','t','U','u','V','v','W','w','X','x','Y','y','Z','z',]
nclasses = len(classes)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)

#Starting the camera
cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  try:
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height,width = gray.shape
    upper_left = (int(width/2-56),int(height/2-56))
    bottom_right = (int(width/2+56),int(height/2+56))
    cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
    roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    im_pil = Image.fromarray(roi)
    Image_bw = im_pil.convert("L")
    Image_bw_resize = Image_bw.resize((28,28),Image.ANTIALIAS)
    Image_bw_resize_inverted = PIL.ImageOps.invert(Image_bw_resize)
    pixel_filter = 20
    min_pixel = np.percentile(Image_bw_resize_inverted,pixel_filter)
    Image_bw_resize_inverted_scale = np.clip(Image_bw_resize_inverted-min_pixel,0,255)
    max_pixel = np.max(Image_bw_resize_inverted)
    Image_bw_resize_inverted_scale = np.asarray(Image_bw_resize_inverted_scale)/max_pixel
    test_sample = np.array(Image_bw_resize_inverted_scale).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print("predicted: ",test_pred)
    cv2.imshow("frame",gray)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break
  except Exception as e:
    pass
cap.release()
cv2.destroyAllWindows()
