#Canvas based on simple paint application by https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

import numpy as np
from tkinter import *
from PIL import ImageGrab
from PIL import Image
import pickle
from keras.models import load_model
from pathlib import Path
from scipy import ndimage
import cv2
import imutils
from skimage.transform import resize

class Paint(object):

    DEFAULT_PEN_SIZE = 20

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.clear_button = Button(self.root, text='clear', command=self.use_clear)
        self.clear_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.canvas = Canvas(self.root, bg='white', width=280, height=280)
        self.root.resizable(False, False)
        self.canvas.grid(row=1, columnspan=5)

        #creates a Label to write the classification results
        self.label = Label(self.root, text='               ', font=("Helvetica", 20))
        self.label.grid(row=1, column=400,pady=2, padx=2)
        
        self.setup()

        #loads SVC model
        model_file = Path("digits_recognition.pickle")
        if  model_file.is_file():      
            self.svc_classifier  = pickle.load(open(model_file, 'rb'))
            self.classify_svc_button=Button(self.root, text='svc', command=self.classify_svc)
            self.classify_svc_button.grid(row=0, column=3)                 
            
        #loads keras NN model
        model_file = Path("digits_recognition_cnn.h5")
        if  model_file.is_file():               
            self.keras_classifier = load_model(model_file)
            self.classify_keras_button=Button(self.root, text="keras", command=self.classify_keras)
            self.classify_keras_button.grid(row=0, column=4)               
            
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.color = 'black'
        self.eraser_on = False
        self.active_button = self.pen_button
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def canvas_to_array(self, image_bit_length):
       
        #macete de salvar dados do canvas copiand para uma imagem PIL
        widget=self.canvas
        x=self.root.winfo_rootx() + widget.winfo_x()
        y=self.root.winfo_rooty() + widget.winfo_y()
        x1=x + widget.winfo_width()
        y1=y + widget.winfo_height()

        #shrinks drawn image to the requested size in pixels and converts it from RGB to grayscale "L"
        image = ImageGrab.grab((x+2, y+2, x1-2, y1-2)).convert("L")

        array = np.asarray( image )    # converts image to array      
        array = ~array                          #the bitmap to be fed to model is inverted: black=255, white=0
        blank_image = np.amax(array) == 0
        
        if(not blank_image):          
            #centers and rescales  the image
            array = self.center_of_mass(array)
            array = self.scale_image(array)

        #shrinks drawing to the size requested by the model
        array_resized = resize(array, (image_bit_length, image_bit_length))
        max_number = np.amax(array_resized)
        if(not blank_image):    
            array_resized = array_resized * (255/max_number)   
            array_resized = array_resized.astype(int)

        #saves pre-processed image for debugging
        cv2.imwrite("paint.bmp", array_resized)
                
        return array_resized

    # After drawing a digit in the above area (280x280 pixels), the drawing is rescaled to make its bounding box fit into a 200x200 pixels region 
    def scale_image(self, array):
        thresh = cv2.threshold(array, 25, 255, cv2.THRESH_BINARY)[1]     #transforma imagem para preto e branco com limiar em 25
       
        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        #calcula altura e largura da imagem detectada
        image_width = extRight[0] - extLeft[0]
        if image_width < 1:
            image_width = 1
        image_height = extBot[1] - extTop[1]
        if image_height < 1:
            image_height = 1        

       #calcula zoom necessário para o numero ficar no padrão do datasetMNIST
        x_zoom =  (array.shape[0] * 200/280) / image_width
        y_zoom = (array.shape[1] * 200/280) / image_height
        #print(image_width, image_height, x_zoom, y_zoom)
        
        return self.cv2_clipped_zoom(array,y_zoom)

    #https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    #Center zoom in/out of the given image and returning an enlarged/shrinked view of  the image without changing dimensions
    #Args:img : Image array zoom_factor : amount of zoom as a ratio (0 to Inf)
    def cv2_clipped_zoom(self, img, zoom_factor):    
        height, width = img.shape[:2] # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

        ### Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1,x1,y2,x2])
        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

        result = cv2.resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode='constant')
        assert result.shape[0] == height and result.shape[1] == width
        return result

    #shifts image on both axis so that the center of mass matches the geometric center
    def center_of_mass(self, array):
        #determines center of mass and geometric center
        cm = ndimage.measurements.center_of_mass(array)          
        c1 = ndimage.measurements.center_of_mass(np.ones_like(array))

       #shifts image
        centered_array = np.roll(array, int(c1[0]-cm[0]) , axis=0)
        return np.roll(centered_array, int(c1[1]-cm[1]) , axis=1)

    def show_classification(self, predicted):
        predicted *= 100.0/sum(predicted[0])
        digit = np.argmax(predicted[0])
        acc = max(predicted[0])

        #shows probabilities for each number
        print('\nThe probabilities for this number are:\n')
        for i in range(len(predicted[0])):
            print('%d:      %.1f%%' % (i, predicted[0][i] ))

        #show recognized number in canvas
        self.label.configure(text= "\n\nThe digit is: " + str(digit) +"\n\nAccuracy: "+ str(int(acc)) + "%")    

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_clear(self):
        self.canvas.delete("all")
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        paint_color = 'white' if self.eraser_on else self.color
        
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
                               
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
 
    def classify_keras(self):
        array = self.canvas_to_array(28)     #MNIST images are 28x28
        array = array/255.0   #normalizes
       
        array_flat = array.reshape(1, 28, 28, 1)   #converts to the required CNN shape
        predicted = self.keras_classifier.predict(array_flat)   

        self.show_classification(predicted)


    def classify_svc(self):
        array = self.canvas_to_array(28)     #MNIST images are 28x28
        array = array/255.0   #normalizes
    
        #SVC receives one dimensional arrays 
        array_flat = array.reshape((1, -1))    
        predicted = self.svc_classifier.predict_proba(array_flat)

        self.show_classification(predicted)

if __name__ == '__main__':
    Paint()
