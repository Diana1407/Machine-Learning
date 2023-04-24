import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#in train_data salvez id-urile pozelor alaturi de label-urile corespunzatoare, pe care urmeaza sa le sparg mai jos 
#skiprow - sare primul rand din citire deoarece acolo este headerul (adica: id, class)
train_data = np.loadtxt('C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/train_labels.txt', dtype=str, skiprows=1)
train_labels = []
train_images = []

#parcurg datele dand-ule split pentru a salva iamginile si label-urile separat
for data in train_data:
    train_id, train_label = data.split(',')

    #ma folosesc de modulul PIL pentru citire
    pre_img = PIL.Image.open(f'C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/data2/{train_id}.PNG')
    #pentru a nu modifica array-ul original voi face un deepcopy al imaginii
    #imi convertesc imaginea la tipul np array, totodata dandu-i resize
    img = copy.deepcopy(np.asarray(pre_img.resize((64, 64))))
    train_images.append(img)
    pre_img.close()
    train_labels.append(train_label)
    
train_images = np.array(train_images)
train_labels = np.array(train_labels).astype('float')

#cream instanta train_datagen care are cele 3 tipuri de augmentare de date
train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   zoom_range = 0.1,
                                   horizontal_flip = True)

validation_data = np.loadtxt('C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/validation_labels.txt', dtype=str, skiprows=1)
validation_labels = []
validation_images = []

for data in validation_data:
    validation_id, validation_label = data.split(',')

    pre_img = PIL.Image.open(f'C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/data2/{validation_id}.PNG')
    img = copy.deepcopy(np.asarray(pre_img.resize((64, 64))))
    validation_images.append(img)
    pre_img.close()

    validation_labels.append(validation_label)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels).astype('float')


#din datele de test din fisierul sample_submission avem nevoide doar de imaginile de test nu si de label
test_data = np.loadtxt('C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/sample_submission.txt', dtype=str, skiprows=1)
test_images = []
test_ids = []
for data in test_data:
    test_id, _ = data.split(',')

    pre_img = PIL.Image.open(f'C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/data2/{test_id}.PNG')
    img = copy.deepcopy(np.asarray(pre_img.resize((64, 64))))
    test_images.append(img)
    pre_img.close()

    test_ids.append(test_id)

test_images = np.array(test_images)

#cod pt afisarea imaginilor
# for img in test_images:
#     plt.imshow(img)
#     plt.show()

#cream modelul de tipul secvential si adaugam pe rand layerele pe care le folosim

# model = models.Sequential()
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.45))
# model.add(layers.Dense(1, activation='sigmoid'))
#prima varinata a modelului => 0.53

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.45))
model.add(layers.Dense(1, activation='sigmoid'))
# a doua varianta a modelului

#dam reshape la imagini deoarece modelele au nevoie de 4D shape
train_images = train_images.reshape(-1, 64, 64, 3)
validation_images = validation_images.reshape(-1, 64, 64, 3)
test_images = test_images.reshape(-1, 64, 64, 3)

#compilez modelul 
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy']) 

#antrenez modelul, doar ca prima data fara datele augmentate           
# history = model.fit(train_images/255, train_labels, epochs=30, batch_size = 64, validation_data=(validation_images/255, validation_labels))
# => 0.60

#antrenez modelul iar cu ajutorul lui flow ne generam noi imagini augmentate pentru setul de date
#impart imaginile de train la 255 deoarece le aplicam o normalizare
history = model.fit(train_datagen.flow(train_images/255, train_labels, batch_size=64), epochs=20, validation_data=(validation_images/255, validation_labels))
# => 0.649

#aici afisez graficele pentru loss pe train si validation
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#aici afisez graficele pentru accuracy pe train si validation
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

#se face predict pe imaginile de test, care si acestea sunt normalizate
pred = model.predict(test_images/255)

#conform regulilor scriem in fisierul de ieisre outpurul, incepand cu antetul id,class
#rurmat pe fiecare rand de id-ul imaginii de test si label-ul corespunzator
f= open("Submisie_CNN_testaug_testt.csv", "w")
f.write("id,class")
f.write('\n')
for i in range(len(test_ids)):
    f.write(test_ids[i])
    f.write(',')
    #fiind un CNN care se termina cu un layer Dense sigmoid, va genera numere in intervalul [0,1]
    #astfel ca trebuie comparat cu un numar din interval pentru a delimita de ce categorie apartine
    if(pred[i] > 0.4):
        f.write("1")
    else:
        f.write("0")
    f.write('\n')