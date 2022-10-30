#importation des bibliothèques requises

import numpy as np
import cv2
from keras.models import load_model



#forme d'entrée sur laquelle nous avons formé notre modèle

input_shape = (120,120,3)
labels_dict = {0: 'WithMask', 1: 'WithoutMask'}
color_dict = {0 : (0,255,0), 1:(0,0,255)} #if 1 - RED color, 0 - GREEN color
model = load_model('modemask.hdf5')


from mtcnn.mtcnn import MTCNN #importer mtcnn
detector = MTCNN()



size = 4
webcam = cv2.VideoCapture(0)  # Utiliser la caméra 0 - webcam par défaut

#
while True: #nous lisons image par image
    (rval, im) = webcam.read()
    # im = cv2.flip(im, 1, 1)  # Flip to act as a mirror
#
#     # Redimensionner l'image pour accélérer la détection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    rgb_image = cv2.cvtColor(mini, cv2.COLOR_BGR2RGB) # MTCNN a besoin du fichier au format RVB, mais cv2 lit au format BGR. Nous convertissons donc.
    faces = detector.detect_faces(mini) # détecter les visages ---> nous aurons les coordonnées (x,y,w,h)


#
#     # Dessinez des rectangles autour de chaque face
    for f in faces:
        x, y, w, h = [v * size for v in f['box']]


#         # cropping the face portion from the entire image
        face_img = im[y:y + h, x:x + w]
        # print(face_img)
        resized = cv2.resize(face_img, (input_shape[0],input_shape[1])) # redimensionner l'image à la taille d'entrée requise sur laquelle nous avons formé notre modèle

        reshaped = np.reshape(resized, (1, input_shape[0],input_shape[1], 3)) # nous avons utilisé ImageDatagenerator et nous avons formé notre modèle par lots
                                                                        # # donc la forme d'entrée de notre modèle est (batch_size,height,width,color_depth)
                                                                        # nous convertissons l'image dans ce format. i.e. (height,width,color_depth) ---> (batch_size,height,width,color_depth)

        result = model.predict(reshaped) #predicting
#         # print(result)
#
        label = np.argmax(result, axis=1)[0] #obtenir l'indice pour la valeur maximale
#
        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2) # Boîte englobante (grand rectangle autour du visage)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1) #petit rectangle au-dessus de BBox où nous mettrons notre texte
                                                                        #Une épaisseur de -1 px remplira la forme du rectangle avec la couleur spécifiée.
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     # affiche limage
    cv2.imshow('LIVE FACE MASK  DETECTION', im)
    key = cv2.waitKey(10)
#     # si la touche Esc est enfoncée, sortez de la boucle
    if key == 27:  # The Esc key
        break
# # arret web cam
webcam.release()
#
# #Fermez toutes les fenêtres ouvertes
cv2.destroyAllWindows()

