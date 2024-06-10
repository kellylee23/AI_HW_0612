## SGD 이용 - 모델컴파일 (1) - 기존모델

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD

# 데이터 로드
(data_train, data_validation, data_test), ds_info = tfds.load(
    'cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]','train[80%:]'], 
    with_info=True, as_supervised=True
    )

# 데이터 전처리 함수
def resizing(image, label):
    image = tf.image.resize(image, (160, 160))
    image = (image / 127.5) -1 
    return image, label

# 데이터셋 전처리
batch_size = 32
train = data_train.map(resizing).shuffle(500).batch(batch_size)
validation = data_validation.map(resizing).batch(batch_size)
test = data_test.map(resizing).batch(batch_size)

model = Sequential()
model.add(Conv2D(16,(3,3), padding='same', activation='relu', input_shape=(160, 160, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

# 모델 컴파일
model.compile(loss='binary_crossentropy',optimizer=SGD(),metrics=['accuracy'])

# 모델 훈련
history = model.fit(
    train,
    validation_data=validation,
    epochs=10
)

# 모델 평가
loss, accuracy = model.evaluate(test, verbose=0)
print('Loss:', loss)
print('Accuracy:', accuracy)

# 모델 예측
for images, labels in test.take(1):  # 테스트 데이터에서 1개의 배치 가져오기
    predictions = model.predict(images)
    for i in range(len(images)):
        predicted_label = 'Cat' if predictions[i] < 0.5 else 'Dog'
        true_label = 'Cat' if labels[i] == 0 else 'Dog'
        
        # 이미지와 예측 결과를 출력
        plt.imshow((images[i] + 1) / 2)  # 이미지 정규화 해제
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')
        plt.show()