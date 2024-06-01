import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import torch

# Desactivar la verificación del certificado SSL para MacOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Cargar del modelo YOLOv5 utilizando torch.hub.load()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Función para procesar cada fotograma del video
def process_frame(frame):
    # Detectar objetos en el fotograma
    results = model(frame)

    # Filtrar deteccion para carros
    car_detections = [detection[:4] for detection in results.pred[0] if detection[-1].item() == 2]

    # Dibujar cuadros
    for detection in car_detections:
        x1, y1, x2, y2 = detection
        color = (255, 0, 0)  # Color verde
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Agregar texto indicando que es un carro dentro del cuadro detectado
        cv2.putText(frame, 'Carro', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Ventana con Tkinter
def show_video():
    ret, frame = cap.read()  # Leer el fotograma
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Redimensionar el fotograma
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a formato RGB para mostrar en Tkinter
        frame = process_frame(frame)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)
        label_video.after(10, show_video) # Actualizar ventana despuéss de 5ms

# Comenzar video
cap = cv2.VideoCapture('video/video.mp4')
if not cap.isOpened():
    print("Error al abrir el archivo de video")
    exit()

# Comenzar ventana Tkinter
root = tk.Tk()
root.title("Detección de coches")
label_video = tk.Label(root)
label_video.pack()

# Mostrar el video en la ventana de Tkinter
show_video()

# Ejecutar el bucle principal de Tkinter
root.mainloop()

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
