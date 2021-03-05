import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
argum = argparse.ArgumentParser()
argum.add_argument("--mode",help="train/display")
mode = argum.parse_args().mode

# plots accuracy and loss curves
def plot_model_history(model):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axis = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axis[0].plot(range(1,len(model.history['accuracy'])+1),model.history['accuracy'])
    axis[0].plot(range(1,len(model.history['val_accuracy'])+1),model.history['val_accuracy'])
    axis[0].set_title('Model Accuracy')
    axis[0].set_ylabel('Accuracy')
    axis[0].set_xlabel('Epoch')
    axis[0].set_xticks(np.arange(1,len(model.history['accuracy'])+1),len(model.history['accuracy'])/10)
    axis[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axis[1].plot(range(1,len(model.history['loss'])+1),model.history['loss'])
    axis[1].plot(range(1,len(model.history['val_loss'])+1),model.history['val_loss'])
    axis[1].set_title('Model Loss')
    axis[1].set_ylabel('Loss')
    axis[1].set_xlabel('Epoch')
    axis[1].set_xticks(np.arange(1,len(model.history['loss'])+1),len(model.history['loss'])/10)
    axis[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train = 'data/train'
val = 'data/test'

no_of_train = 28709
no_of_val = 7178
batch_size = 64
no_of_epoch = 50

train_datagenerator = ImageDataGenerator(rescale=1./255)
val_datagenenerator = ImageDataGenerator(rescale=1./255)

train_generator = train_datagenerator.flow_from_directory(
        train,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagenerator.flow_from_directory(
        val,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
mdl = Sequential()

mdl.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
mdl.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
mdl.add(MaxPooling2D(pool_size=(2, 2)))
mdl.add(Dropout(0.25))
mdl.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
mdl.add(MaxPooling2D(pool_size=(2, 2)))
mdl.add(Dropout(0.25))

mdl.add(Flatten())
mdl.add(Dense(1024, activation='relu'))
mdl.add(Dropout(0.5))
mdl.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this
if mode == "train":
    mdl.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    mdl_info = mdl.fit_generator(
            train_generator,
            steps_per_epoch=no_of_train // batch_size,
            epochs=no_of_epoch,
            validation_data=validation_generator,
            validation_steps=no_of_val // batch_size)
    plot_mdl_history(model_info)
    mdl.save_weights('model.h5')

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    mdl.load_weights('model.h5')

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    capturevideo = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = capturevideo.read()
        if not ret:
            break
        casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = casc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = mdl.predict(crop_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capturevideo.release()
    cv2.destroyAllWindows()