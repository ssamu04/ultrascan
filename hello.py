
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

SIZE_X = 768
SIZE_Y = 512

train_images = []
train_masks = []

image_paths = sorted(glob.glob("training_set/*_HC.png"))
for img_path in image_paths:
    # print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    train_images.append(img)

mask_paths = sorted(glob.glob("training_set/*_HC_Annotation.png"))
for img_path in mask_paths:
    # print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    train_masks.append(img)

train_images = np.array(train_images)
train_masks = np.array(train_masks)

X = train_images.astype('float32')
Y = (train_masks > 0.5).astype('float32')
Y = Y[:, :, :, 0]

# #print statements for debugging
# print("X shape:",X.shape)
# print("Y shape:",Y.shape)

# unique_values = np.unique(Y)
# print("Unique values in train_masks:", unique_values)
# # Verify if the masks are binary
# if set(unique_values).issubset({0.0, 1.0}):
#     print("train_masks are binary.")
# else:
#     print("train_masks are not binary.")

# #used to show images
# num_images = 5
# plt.figure(figsize=(10, 5))
# for i in range(num_images):
#     # Display image
#     plt.subplot(2, num_images, i + 1)
#     plt.imshow(X[i] / 255.0)  # Scale the image to [0, 1] for display
#     plt.title(f"Image {i+1}")
#     plt.axis('off')

#     # Display corresponding mask
#     plt.subplot(2, num_images, i + 1 + num_images)
#     plt.imshow(Y[i], cmap='gray')
#     plt.title(f"Mask {i+1}")
#     plt.axis('off')
# plt.show()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# preprocess input
x_train = preprocess_input(x_train) / 255.0
x_val = preprocess_input(x_val) / 255.0

# define model
model = sm.Linknet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

#print(model.summary())

history=model.fit(x_train,
            y_train,
            batch_size=8, 
            epochs=10,
            verbose=1,
            validation_data=(x_val, y_val))

#plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#model.save('membrane.h5')


# from tensorflow import keras
# model = keras.models.load_model('membrane.h5', compile=False)
# #Test on a different image
# #READ EXTERNAL IMAGE...
# test_img = cv2.imread('membrane/test/0.png', cv2.IMREAD_COLOR)       
# test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
# test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# test_img = np.expand_dims(test_img, axis=0)

# prediction = model.predict(test_img)

# #View and Save segmented image
# prediction_image = prediction.reshape(mask.shape)
# plt.imshow(prediction_image, cmap='gray')
# plt.imsave('membrane/test0_segmented.jpg', prediction_image, cmap='gray')


