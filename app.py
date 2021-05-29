from flask import Flask,redirect,render_template,url_for

import cv2
import numpy as np
from pyzbar.pyzbar import decode

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post','get'])
def predict():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
 
    while True:
        success, img = cap.read()
        for barcode in decode(img):
            myData = barcode.data.decode('utf-8')
            pts = np.array([barcode.polygon],np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(255,0,255),5)
            pts2 = barcode.rect
            cv2.putText(img,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,255),2)
 
        cv2.imshow('Result',img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    return render_template('index.html')
    
if __name__=='__main__':
    app.run(debug=True)