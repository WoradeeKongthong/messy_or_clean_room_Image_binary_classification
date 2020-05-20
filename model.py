# import dataset & do image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('images/train',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('images/val',
                                            target_size=(128,128),
                                            batch_size = 32,
                                            class_mode = 'binary')

# create CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# initialize CNN
clf = Sequential()
# first set of conv+pooling
clf.add(Conv2D(32,(3,3), input_shape=(128,128,3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))
# second set
clf.add(Conv2D(32,(3,3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))
# third set
clf.add(Conv2D(32,(3,3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))
# flattening
clf.add(Flatten())
# full connection
clf.add(Dense(units=256, activation='relu'))
clf.add(Dropout(0.4))
clf.add(Dense(units=128, activation='relu'))
clf.add(Dropout(0.4))
clf.add(Dense(units=64, activation='relu'))
clf.add(Dropout(0.4))
clf.add(Dense(units=1, activation='sigmoid'))

# compiling the CNN
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training the CNN
clf.fit_generator(training_set,
                  steps_per_epoch = 96,
                  epochs = 10,
                  validation_data = test_set,
                  validation_steps = 10)

# check indices
#training_set.class_indices

# making new predictions
import numpy as np
from keras.preprocessing import image
test = []

for i in range(10):
    test_image = image.load_img('images/test/'+str(i)+'.png', target_size=(128,128))
    test_image = image.img_to_array(test_image)
    test.append(test_image)
test = np.array(test)

# print prediction
result = clf.predict(test)
for i in range(len(result)):
    prediction = 'clean' if result[i][0] == 0 else 'messy'
    print('{}.png is {}'.format(i,prediction))

#-----------------------------------------------------------------------------
# save the model with pickle
import pickle

pickl = {
    'model':clf
}
pickle.dump(pickl, open('messyOrNotCnn'+".p","wb"))

# load the model to test
data = pickle.load(open('messyOrNotCnn.p','rb'))
clf_model = data['model']

test_image = image.load_img('images/test/test1.jpg', target_size=(128,128))
test_image = image.img_to_array(test_image)
test_image = np.array([test_image])

result1 = clf_model.predict(test_image)
if result1[0][0]==0:
    print('Your room is clean.')
else:
    print('Your room is messy.')