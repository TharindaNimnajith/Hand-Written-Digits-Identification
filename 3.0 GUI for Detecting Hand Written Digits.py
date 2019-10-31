import tkinter as tk
from PIL import Image, ImageDraw
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, svm

digits = datasets.load_digits()

data = digits.data
target = digits.target
images = digits.images

#64 features (pixel values)
#1797 targets

classifier = svm.SVC(gamma = 0.0001, C = 100)
classifier.fit(data, target)

##print(len(target)) #1797
##
##print(data.shape)  #(1797, 64)
##data is a 2D Array
##
##print(data[0])
##[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
## 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
##  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
##  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
##
##plt.imshow(images[1], cmap = 'binary')
##plt.show()

window = tk.Tk()

width  = 500
height = 500
font   = 'Helvetica 20 bold'

def event_function(event):
    #canvas1.create_oval((20, 20, 100, 100), outline = 'red', width = 3)
    #print(event.char)
    #canvas1.create_text(20, 20, text = event.char)
    #label_predict.config(text = event.char)
    #canvas1.create_oval((20, 20, 40, 40), fill = 'black')
    
    x1 = event.x - 30
    y1 = event.y - 30
    x2 = event.x + 30
    y2 = event.y + 30

    canvas1.create_oval((x1, y1, x2, y2), fill = 'black')
    
    global image_draw
    image_draw.ellipse((x1, y1, x2, y2), fill = 'black')
    
def save():
    #canvas1.create_rectangle((20, 20, 100, 100), fill = 'black')
    #canvas1.create_rectangle((20, 20, 100, 100), outline = 'red', width = 5)
    #canvas1.create_oval((20, 20, 100, 150), outline = 'red', width = 3)

    global image, image_array
    image_array = np.array(image)
    image_array = cv2. resize(image_array, (8, 8))
    cv2.imwrite('digits.jpg', image_array)
    plt.imshow(image_array, cmap = 'binary')
    plt.title('Digit')
    plt.savefig('3.0 Digit.png')
    plt.show()
    
def predict():
    global image_array
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_array = cv2.resize(image_array, (8, 8))
    image_array_flat = image_array.ravel() #converting 2D array into 1D array
    image_array_flat = 15 - ((image_array_flat / 255.0) * 15.0)

    result = classifier.predict([image_array_flat])
    label_predict.config(text = 'PREDICTED DIGIT: '+str(result[0]))

def clear():    
    canvas1.delete('all')

    global image, image_draw
    image = Image.new('RGB', (width, height), (255, 255, 255))
    image_draw = ImageDraw.Draw(image)

canvas1 = tk.Canvas(window, width = width, height = height, bg = 'white')
canvas1.grid(row = 0, column = 0, padx = 2, pady = 4, columnspan = 4) #columnspan = 3 -> creates 3 columns in the below row

label_predict = tk.Label(window, text = 'PREDICTED DIGIT: NONE', bg = 'gray80', font = font)
label_predict.grid(row = 1, column = 0, pady = 4, columnspan = 4)

button_clear = tk.Button(window, text = 'Clear', bg = 'black', fg = 'white', font = font, command = clear)
button_clear.grid(row = 2, column = 0, pady = 6, padx = 5)

button_save = tk.Button(window, text = 'Save', bg = 'green', fg = 'white', font = font, command = save)
button_save.grid(row = 2, column = 1, pady = 6, padx = 5)

button_predict = tk.Button(window, text = 'Predict', bg = 'blue', fg = 'white', font = font, command = predict)
button_predict.grid(row = 2, column = 2, pady = 6, padx = 5)

button_exit = tk.Button(window, text = 'Exit', bg = 'red', fg = 'white', font = font, command = window.destroy)
button_exit.grid(row = 2, column = 3, pady = 6, padx = 5)

##widgetName.bind('event', function)
#canvas1.bind('<Button-1>', event_function)
#label_predict.bind('<Button-1>', event_function)
#button_predict.bind('<Button-1>', event_function)
#canvas1.bind('<Enter>', event_function)
#canvas1.bind('<Leave>', event_function)
#canvas1.bind('<Double-Button-1>', event_function) #left mouse button
#canvas1.bind('<Double-Button-2>', event_function) #scroll button
#canvas1.bind('<Double-Button-3>', event_function) #right mouse button
#window.bind('<Key>', event_function)

canvas1.bind('<B1-Motion>', event_function)

image = Image.new('RGB', (width, height), (255, 255, 255))
image_draw = ImageDraw.Draw(image)
