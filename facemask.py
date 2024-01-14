import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import cv2
import datetime

# Sequential API: stack of layers
face_mask_model = Sequential()
face_mask_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
face_mask_model.add(MaxPooling2D())
face_mask_model.add(Conv2D(32, (3, 3), activation='relu'))
face_mask_model.add(MaxPooling2D())
face_mask_model.add(Conv2D(32, (3, 3), activation='relu'))
face_mask_model.add(MaxPooling2D())
face_mask_model.add(Flatten())
face_mask_model.add(Dense(100, activation='relu'))
face_mask_model.add(Dense(1, activation='sigmoid'))

face_mask_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

trained_model = face_mask_model.fit_generator(
    training_set,
    epochs=10,
    validation_data=test_set,
)

face_mask_model.save('face_mask_model.h5', trained_model)

# To test for individual images
loaded_model = load_model('face_mask_model.h5')
test_image = image.load_img(r'C:\Users\91991\OneDrive\Desktop\DSA_DA\FaceMaskDetector\train\with_mask\15-with-mask.jpg',
                             target_size=(150, 150, 3))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
loaded_model.predict(test_image_array)[0][0]

# IMPLEMENTING LIVE DETECTION OF FACE MASK
loaded_model = load_model('face_mask_model.h5')
video_capture = cv2.VideoCapture(0)
face_cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while video_capture.isOpened():
    _, frame = video_capture.read()
    detected_faces = face_cascade_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in detected_faces:
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)
        test_img = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_img_array = image.img_to_array(test_img)
        test_img_array = np.expand_dims(test_img_array, axis=0)
        prediction = loaded_model.predict(test_img_array)[0][0]
        if prediction == 1:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame, 'NO MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, 'MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        current_datetime = str(datetime.datetime.now())
        cv2.putText(frame, current_datetime, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
