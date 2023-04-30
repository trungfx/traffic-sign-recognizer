from tkinter import *
from tkinter import ttk
import tkinter 
import cv2
import PIL.Image,PIL.ImageTk
import threading
import os
from tkinter import filedialog
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
seg=SelfiSegmentation()
from PIL import Image,ImageTk

#------------------------------------------------

def openf():
    return filedialog.askopenfilename()

# ------------------------------------------------
    
def xulycamera():
    global cap
    cap=cv2.VideoCapture(0)
    XLHA()
    
# ------------------------------------------------

def xulyfile():
    global cap
    cap=cv2.VideoCapture(openf())
    XLHA()
    
 # ------------------------------------------------
 
a=threshold = 0.9

# ------------------------------------------------

def majority_element(num_list):
    idx, ctr = 0, 1
        
    for i in range(1, len(num_list)):
        if num_list[idx] == num_list[i]:
            ctr += 1
        else:
            ctr -= 1
            if ctr == 0:
                idx = i
                ctr = 1
        
    return num_list[idx]

# ------------------------------------------------

def preprocess_img(imgBGR, erode_dilate=True):  
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)
        
    img_bin = np.maximum(img_Bbin, img_Bbin)
    
    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=1)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=1)

    return img_bin

# ------------------------------------------------

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


#-------------------------------------------------
window= Tk()
window.title("Hệ thống nhận diện biển báo giao thông")
# icon tieu de
icon = tkinter.PhotoImage(file = 'icon/icon_CTU.png') 
resize_icon=icon.subsample(10, 10)
window.iconphoto(True, resize_icon)
window.geometry('610x580+600+50')#kich thuoc mang hinh va vi tri xuat hien
w=600
h=550

# ===============================================================
cap = cv2.VideoCapture("./video/5.mp4")

# ===============================================================

vien_chung=Canvas(window, bg='#FFFFFF',width=w, height=h)
vien_chung.place(relx=0, rely=0)
vien_chung.create_rectangle(10,4,w,h-3,outline="black",width=3)

# ------------------------------------------------

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
import tensorflow  as tf
from keras.preprocessing import image
from numpy import argmax

model=keras.models.load_model("trafficsign.h5", compile=False)
categories =['khong BB','Cam','Chi dan','Hieu lenh','Nguy hiem']

# ---------------------------------------------- 
def XLHA():
    threadOfFrame = threading.Thread(target=frame)
    threadOfFrame.start() 
stop_threads=False   

# ------------------------------------------------

def frame():
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Majority=[]
    Ten=""
    dem=0
    while True:
        try:
            ret , img=cap.read()
            black=(0,0,0)
            image1=seg.removeBG(img,black,threshold=0.7)
            img_bin = preprocess_img(img, False)
            min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
            rects = contour_detect(img_bin, min_area=min_area)  
            for rect in rects:
                xc = int(rect[0] + rect[2] / 2)
                yc = int(rect[1] + rect[3] / 2)

                size = max(rect[2], rect[3])
                x1 = max(0, int(xc - size / 2))
                y1 = max(0, int(yc - size / 2))
                x2 = min(cols, int(xc + size / 2))
                y2 = min(rows, int(yc + size / 2))

                if rect[2] > 35 and rect[3] > 35:            
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                crop_img = np.asarray(image1[y1:y2, x1:x2])   
                            
                image_nhandang = crop_img
                image_nhandang = cv2.resize(image_nhandang, (64, 64))  
                image_nhandang = np.array(image_nhandang, dtype="float") / 255.0            
                image_nhandang=np.expand_dims(image_nhandang, axis=0)
                print(image_nhandang.shape)
                pred=model.predict(image_nhandang)
                Res=argmax(pred,axis=1)
                Majority.append(Res[0])
                dem = dem +1
                if Res >= a: 
                    if dem > 10:
                        dem=0
                        Ten="{0}".format(categories[majority_element(Majority)])
                        Majority=[]
                        # ------------------------------------------------
                        doikichthuoc = cv2.resize(img, (300, 260))
                        rgb  =cv2.cvtColor(doikichthuoc,cv2.COLOR_BGR2RGB)              
                        img5 = Image.fromarray(rgb)
                        pic = ImageTk.PhotoImage(img5)
                        bienbao.configure(image=pic)
                        # ------------------------------------------------
                        ten.config(text =Ten)
                        # ------------------------------------------------
                        cv2.putText(img,Ten, (rect[0], rect[1]),cv2.FONT_HERSHEY_PLAIN, 2, (255,0 ,0), 2)
            # cv2.imshow("video_traffic_sign",img)
            if cv2.waitKey(10) & 0xFF == ord('q'):                          
                break
            if stop_threads:                             
                break
        except:
            continue
# ---------------------------------------------- 
def tat():
    global stop_threads
    stop_threads=True
# ==============================================


he_thong=Label(window, text="HỆ THỐNG NHẬN DIỆN BIỂN BÁO \n GIAO THÔNG VIỆT NAM",
               font = ("Times New Roman", 14, "bold"),fg='#3f609a',bg='#FFFFFF')
he_thong.place(relx=0.5,rely=0.07, anchor=N)

# -----------------------------------------------------
he_thong=Label(window, text="-------------o00o-------------",
               font = ("Times New Roman", 10, "bold"),fg='#3f609a',bg='#FFFFFF')
he_thong.place(relx=0.5,rely=0.17, anchor=N)

# -----------------------------------------------

webcam=PIL.Image.open("icon/webcam.png")
resize_webcam=webcam.resize((20,20),PIL.Image.ANTIALIAS)
imgwebcam=PIL.ImageTk.PhotoImage(resize_webcam)
# nut button
button_cam=Button(window, text="Mở camera",width= "95",height="20",command=xulycamera,image=imgwebcam,compound = RIGHT, 
    font = ("Times New Roman", 13),
    bg='green',
    fg='white',
    activebackground='#00abfd',
    activeforeground='white')
button_cam.place(relx=0.07,rely=0.45, anchor=SW)

# ---------------------------------------------

file=PIL.Image.open("icon/folder.png")
resize_file=file.resize((20,20),PIL.Image.ANTIALIAS)
imgfile=PIL.ImageTk.PhotoImage(resize_file)

buttonmofile=Button(window, text="Mở file ",width= "95",height="20",command=xulyfile,image=imgfile,compound = RIGHT, 
    font = ("Times New Roman", 13),
    bg='gray',
    fg='white',
    activebackground='#00abfd',
    activeforeground='white')
buttonmofile.place(relx=0.07,rely=0.55, anchor=SW)
#----------------------------------------------

dong=PIL.Image.open("icon/close.png")
resize_dong=dong.resize((20,20),PIL.Image.ANTIALIAS)
imgdong=PIL.ImageTk.PhotoImage(resize_dong)

button_cam=Button(window, text="Tắt ",width= "95",height="20",command=tat,image=imgdong,compound = RIGHT, 
    font = ("Times New Roman", 13),
    bg='white',
    fg='black',
    activebackground='#00abfd',
    activeforeground='white')
button_cam.place(relx=0.07,rely=0.65, anchor=SW)

# =============================================================

def ExitWindow():
    global stop_threads
    stop_threads = True  
    window.destroy()

button_tat=Button(window, text="Đóng ",width= "95",height="20",command=ExitWindow,image=imgdong,compound = RIGHT,
    font = ("Times New Roman", 13),
    bg='red',
    fg='white',
    activebackground='#df322b',
    activeforeground='white')
button_tat.place(relx=0.07,rely=0.75, anchor=SW)


#----------------------------------------------

ketqua=Label(window, text = "Kết quả:",
          font = ("Times New Roman", 18, "bold"),fg='black',bg='#FFFFFF')
ketqua.place(relx=0.5, rely=0.23, anchor=NW)

#----------------------------------------------

ketquanhip=Label(window, text = "Biển báo:",
          font = ("Times New Roman", 13, "bold"),fg='black',bg='#FFFFFF')
ketquanhip.place(relx=0.4, rely=0.28, anchor=NW)

# ----------------------------------------------------

bienbao=Label(window, text="Ảnh biển báo",
          font = ("Times New Roman", 14),fg='red',bg='#FFFFFF')
bienbao.place(relx=0.37, rely=0.33, anchor=NW)

# ------------------------------------------------
ketqua=Label(window, text = "Tên Biển báo:",
          font = ("Times New Roman", 13, "bold"),fg='black',bg='#FFFFFF')
ketqua.place(relx=0.4, rely=0.8, anchor=NW)
# -----------------------------------------------

ten=Label(window, text = "KQ Tên biển báo:",
          font = ("Times New Roman", 18),fg='red',bg='#FFFFFF')
ten.place(relx=0.45, rely=0.85, anchor=NW)
# ------------------------------------------
ten1=Label(window, text = "Đồ án nhóm 7 CT210",
          font = ("Times New Roman", 10, "italic"),fg='black',bg='#f0f0f0')
ten1.place(relx=0.5, rely=0.99, anchor=S)
window.resizable(False,False)# co dinh kich thuoc mang hinh
window.mainloop()

