import cv2
#from keras.models import load_model
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import image
#import matplotlib.pyplot as plt
import numpy as np
import time
#import RPi.GPIO as GPIO

#Setup del output de la Raspberry
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(22,GPIO.OUT)


###############################################################
###############################################################
#Escoger con qué se quiere correr
#Con tensorflow

#import tensorflow as tf
#animal_model = tf.lite.Interpreter(model_path = '../animal_detector.tflite')

model1 = '../animal_detector.tflite'
model2 = '../animal_detector2.tflite'
model3 = '../animal_detector3.tflite'


#Con tensorflow lite
import tflite_runtime.interpreter as tflite
animal_model = tflite.Interpreter(model_path = model1)


###############################################################
###############################################################


#Se importa el modelo

animal_model.allocate_tensors()

#Obtener inputs y outputs del modelo
model_input = animal_model.get_input_details()
model_output = animal_model.get_output_details()
#print(model_input[0]['shape'])
#print('\n')
#print(model_output)


#Función para preprocesar una imagen dado un path
def image_format(path):
    img = cv2.imread(path)
    #print('Original Dimensions : ',img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height = 180
    img_width = 180
    img = cv2.resize(img, (img_height,img_width), interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ',img.shape)
    #plt.imshow(img)
    img = img.astype(np.float32)
    img = [img]
    img = np.asarray(img)
    #print('Tensor Dimensions : ',img.shape)
    #print(img)
    return img


#Funcion para preprocesar una imagen dado un frame
def image_format2(img):
    #print('Original Dimensions : ',img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height = 180
    img_width = 180
    img = cv2.resize(img, (img_height,img_width), interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ',img.shape)
    #plt.imshow(img)
    img = img.astype(np.float32)
    img = [img]
    img = np.asarray(img)
    #print('Tensor Dimensions : ',img.shape)
    #print(img)
    return img

#Clases en el orden correspondiente
animal_classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'frogs', 'horse', 'sheep', 'spider', 'squirrel']

#Función que predice que animal es dado un tensor de entrada
#Tamaño de entrada: (1,180,180,3)
def predict(tensor):
    animal_model.set_tensor(model_input[0]['index'],tensor)
    animal_model.invoke()
    #Se hace la predicción
    prediction = animal_model.get_tensor(model_output[0]['index'])
    animal_label = animal_classes[prediction.argmax()]
    print('La predicción es que el animal es: ' + animal_label)
    return animal_label

#Ejemplo de prediccion con path
#path = '../animal_classifier/dataset/train/frogs/080_0062.jpg'
#img = image_format(path)
#prediccion = predict(img)
#print('La predicción es que el animal es: ' + prediccion)


#Función que enciende el LED
def detected_frogs():
    GPIO.output(22,True)
    time.sleep(5)
    GPIO.output(22,False)

#Función para predicciones con cámara
def camera_main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError('No se puede abrir la cámara')
    while True:
        try:
            ret,frame = cap.read()
            #gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #animals = faceCascade.detectMultiScale(gray,1.1,4)
            #animals = cv2.CascadeClassifier(args["cascade"])
            #for(x,y,w,h) in animals:
            #cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
            #pos = (x,y)
            
            #Predicción
            img = image_format2(frame)    
            prediccion = predict(img)  
            if(prediccion =='frogs'):
                detected_frogs()
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame,prediccion,(100,100), font, 3, (0,0,255), 2, cv2.LINE_4)
              
                
            cv2.imshow('Camara',frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
                
        except (ValueError) as err:
            #print(err)
            pass
    cap.release()
    cv2.destroyAllWindows()


#Se inicia la aplicación con cámara
camera_main()









