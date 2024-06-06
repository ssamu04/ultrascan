
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

file = "test_set/003_HC.png"
model = keras.models.load_model('my_model.h5', compile=False)
test_img = cv2.imread(file, cv2.IMREAD_COLOR)
original_height, original_width, _ = test_img.shape

test_img = cv2.resize(test_img, (256, 256))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)
prediction_img = prediction.reshape(256, 256)
prediction_img = np.expand_dims(prediction_img, axis=-1)

# resize image back to original shape before displaying
test_img = cv2.resize(test_img[0], (original_width, original_height))
prediction_img = cv2.resize(prediction_img, (original_width, original_height))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(test_img)
ax[0].set_title('Test Image')
ax[1].imshow(prediction_img, cmap='gray')
ax[1].set_title('Prediction')
plt.show()

base_filename = file.split('/')[-1].split('.')[0]
new_filename = f"save/{base_filename}_segmented.png"
plt.imsave(new_filename, prediction_img, cmap='gray')
