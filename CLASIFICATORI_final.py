import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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
    #imi convertesc imaginea la tipul np array, totodata dandu-i resize, folosind metoda NEAREST care copiaza valoarea
    #pixel-ului cel mai apropiat din imaginea originala in cea redimensionata
    img = copy.deepcopy(np.asarray(pre_img.resize((64, 64), resample=PIL.Image.NEAREST)).flatten())
    train_images.append(img)
    pre_img.close()

    train_labels.append(train_label)

#convertesc toate datele de antrenare in obiecte de tip numpy array
train_images = np.array(train_images)
#in plus la label-uri, mi le salvez de tipul float
train_labels = np.array(train_labels).astype(float)

validation_data = np.loadtxt('C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/validation_labels.txt', dtype=str, skiprows=1)
validation_labels = []
validation_images = []

for data in validation_data:
    validation_id, validation_label = data.split(',')

    pre_img = PIL.Image.open(f'C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/data2/{validation_id}.PNG')
    img = copy.deepcopy(np.asarray(pre_img.resize((64, 64), resample=PIL.Image.NEAREST)).flatten())
    validation_images.append(img)
    pre_img.close()

    validation_labels.append(validation_label)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels).astype(float)

#din datele de test din fisierul sample_submission avem nevoide doar de imaginile de test nu si de label
test_data = np.loadtxt('C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/sample_submission.txt', dtype=str, skiprows=1)
test_images = []
test_ids = []
for data in test_data:
    test_id, _ = data.split(',')

    pre_img = PIL.Image.open(f'C:/Users/diana/OneDrive/Documents/Facultate/Anul II, Sem II/IA/ML/Competitie/data/data2/{test_id}.PNG')
    img = copy.deepcopy(np.asarray(pre_img.resize((64, 64), resample=PIL.Image.NEAREST)).flatten())
    test_images.append(img)
    pre_img.close()

    test_ids.append(test_id)

test_images = np.array(test_images)

#aici imi combin datele de train cu cele de validare pentru a putea antrena modelul pe un set mai mare de date
combo_images = []
combo_images.extend(train_images)
combo_images.extend(validation_images)
combo_labels = []
combo_labels.extend(train_labels)
combo_labels.extend(validation_labels)

#creez gridul de parametrii cu diferite valori care dupa vor fi testate pentru a vedea care este cel mai bun
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10]
# }

#formez un obiect de tipul GridSearchCV care cauta hiperparametrii optimi pentru un model
# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(), #modelul folosit
#     param_grid=param_grid, #grid-ul de parametrii
#     cv=5, #nr de subseturi in care impartim datele de antrenare (cross-validation)
#     n_jobs=-1, #nr de nuclee CPU utilizate pentru cautarea in grid in paralel
#     verbose=2 #nivelul de detaliu al informatiilor din timpul cautarii, respectiv va afisa parametrii si scorul acestora
# )

#aplic fit() pe obiectul de tip GridSearchCV, care ajusteaza modelul pe toate combinatiile de parametrii 
#si selecteaza la final cea mai buna combinatie de parametrii
# grid_search.fit(combo_images, combo_labels)

#afisez cei mai buni parametrii si cel mai bun scor
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

#aici apelez functiile pentru clasifiacarea modelului
#cls = GaussianNB()
#cls = RandomForestClassifier()
cls = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=10)
#antrenz doar pe imaginilie de antrenare
#cls.fit(train_images, train_labels)

#antrenez pe imaginile de tarin + validare
cls.fit(combo_images, combo_labels)


#pentru raportul de clasificare
# pred = cls.predict(validation_images)
# print(classification_report(validation_labels, pred))

#afisarea matricei de confuzie
# conf_mat = confusion_matrix(validation_labels, pred, labels = cls.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix = conf_mat,
#                               display_labels = cls.classes_)
# disp.plot()
# plt.show()

#prezic dupa pe imaginile de test
pred = cls.predict(test_images)

#la final afisez in fisier conform regulilor si anume antetul id,class 
#urmat pe fiecare linie de id-ul imaginii si label-ul corespunzator

#f= open("Submisie_NB", "w")
f= open("Submisie_RF_best_param.csv", "w")
f.write("id,class")
f.write('\n')
for i in range(len(test_ids)):
    f.write(test_ids[i])
    f.write(',')
    f.write(str(int(pred[i])))
    f.write('\n')




