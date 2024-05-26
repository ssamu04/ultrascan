
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

SIZE_X = 256
SIZE_Y = 256

# train_images = []
# for directory_path in glob.glob("training_set"):
#     for img_path in glob.glob(os.path.join(directory_path, "*_HC.png")):
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (SIZE_Y, SIZE_X))
#         train_images.append(img)     
# train_images = np.array(train_images)

# train_masks = []
# for directory_path in glob.glob("training_set"):
#     for img_path in glob.glob(os.path.join(directory_path, "*_HC_Annotation.png")):
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (SIZE_Y, SIZE_X))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         train_masks.append(img)      
# train_masks = np.array(train_masks)

train_images = []
train_masks = []

image_paths = sorted(glob.glob("training_set/*_HC.png"))
for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    train_images.append(img)

mask_paths = sorted(glob.glob("training_set/*_HC_Annotation.png"))
for img_path in mask_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    train_masks.append(img)

# Display the first 10 images and masks
num_images_to_display = 10
fig, axes = plt.subplots(num_images_to_display, 2, figsize=(15, 6))
for i in range(num_images_to_display):
    axes[i, 0].imshow(train_images[i+50])
    axes[i, 0].axis('off')
    axes[i, 1].imshow(train_masks[i+50])
    axes[i, 1].axis('off')
plt.tight_layout()
plt.show()


train_images = np.array(train_images)
train_masks = np.array(train_masks)

X = train_images.astype('float32')
Y = train_masks.astype('float32')
Y = np.expand_dims(Y, axis=-1)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)


# define model
model = sm.Linknet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

#print(model.summary())

print("X1 shape:",x_train.shape)
print("Y1 shape:",y_train.shape)

history=model.fit(x_train,
            y_train,
            batch_size=8, 
            epochs=10,
            verbose=1,
            validation_data=(x_val, y_val))


# #accuracy = model.evaluate(x_val, y_val)
#plot the training and validation accuracy and loss at each epoch
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


