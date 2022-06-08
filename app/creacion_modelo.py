import cv2
import matplotlib.pyplot as plt

def show_img(path):
    print('Empiezo a leer la imagen con CV2:')
    img = cv2.imread(path)
    print('Imagen leida con CV2')
    print('Empiezo a mostrar la imagen:')
    plt.imshow(img)

path = 'C://Users//XPC//Documents//TEC//Taller de Sistemas Embebidos//Proyecto 3//img_animals//raw-img//cane//OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg'
#show_img(path)
img = cv2.imread(path)
plt.imshow(img)
