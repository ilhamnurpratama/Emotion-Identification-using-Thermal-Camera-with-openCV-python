#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:04:08 2020

@author: ilhampc
"""

# Thermography Camera Interface (TCI)

import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
from PyQt5 import (QtCore, QtGui)
from datetime import datetime
from PyQt5.QtCore import (QThread, Qt, pyqtSignal)
import cv2
import pandas as pd
import numpy as np
from pyqtgraph import PlotWidget
import sys
import csv
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# List yang dipake
# List Dahi
Ld = [] # Luminansi Dahi
rrdd = [] # Rerata suhu luasan dahi
rrdb = [] # Rerata suhu dahi baseline
waktud = [] # waktu pengukuran suhu dahi
DelTd = [] # Selisih suhu dahi
# List Hidung
Lh = [] # Luminansi hidung
rrhh = [] # Rerata suhu hidung
rrhb = [] # Rerata suhu hidung baseline
waktuh = [] # waktu pengukuran suhu hidung
DelTh = [] # Selisih suhu hidung
# List pipi kanan
Lkn = [] # Luminansi pipi kanan
rrknn = [] # Rerata suhu pipi kanan
rrknb = [] # Rerata suhu pipi kanan baseline
waktukn = [] # waktu pengukuran pipi kanan
DelTkn = [] # Selisih suhu pipi kanan
# List pipi kirikiri
Lkr = [] # Luminansi pipi kiri
rrkrr = [] # Rerata suhu pipi kiri
rrkrb = [] # Rerata suhu pipi kiri baseline
waktukr = [] # Waktu pengukuran pipi kiri
DelTkr = [] # Selisih suhu pipi kiri
JenisEmosi = [] # List tempat menyimpan emosi yang dirasakan
# List umum
dframe = [] # Indeks penyimpanan
td = 0  # Suhu dahi awal
th = 0 #Suhu hidung awal
tkn = 0 #Suhu kanan awal 
tkr = 0 #Suhu kiri awal
# waktu = waktud, waktuh, waktukn, waktukr
# t = td, th, tkn, tkr
# Variabel regresi
M = 0.05953953 #Gradien Luminansi
C = 25.40318592 # Konstanta Luminansi
# SigmaT = temd, temh, temkn, temkr
# JumlahT = len(temd), len(temh), len(temkn), len(temkr) 
# r = rrd,rrh,rrkn,rrkr
# DeltaT = DTd, DTh, DTkn, DTkr
# img_matrix = print(img)
# gray_matrix = print(gray)
# Citra_gray = th = Thread(self)
# th.changePixmap.connect(self.setImage)
# Posisi bagian wajah
'''
wajah = faces
wajahx = faces[0]
wajahy = faces[1]
wajahw = faces[2]
wajahh = faces[3]
dahi= gray[y+10:y+h-130,x+30:x+w-30]
hidung= gray[y+80:y+h-70,x+80:x+w-80,0]
kanan = gray[y+100:y+h-50,x+w-80:x+w-30]
kiri = gray[y+100:y+h-50,x+30:x+80]
'''
# Pixel_Luminansi_Dahi
# print(colorsLd)
# Perkalian gradien dengan Luminansi dahi
# print(nld)
# TDahi = print(temd)
# TDahiReratat60 = print(rrdbm)   
# Pixel_Luminansi_Hidung
# print(colorsLh)
# Perkalian gradien dengan Luminansi hidung
# print(nlh)
# THidung = print(temh)
# THidungReratat60 = print(rrhbm)   
# Pixel_Luminansi_Kanan
# print(colorsLkn)
# Perkalian gradien dengan Luminansi kanan
# print(nlkn)
# TPKanan = print(temkn)
# TPKananReratat60 = print(rrknbm)   
# Pixel_Luminansi_Kiri
# print(colorsLkr)
# Perkalian gradien dengan Luminansi kiri
# print(nlkr)
# TPKiri = print(temkr)
# TPKiriReratat60 = print(rrkrbm) 
# colorsL = colorsLd, colorsLh, colorsLkn, colorLkr
# L = Ld, Lh, Lkn, Lkr
# rr = rrdd, rrhh, rrknn,rrkrr
# rrbm = rrdb, rrhb, rrknb,rrkrb 
ID = [] # List ID
lengkap=[] # List ID lengkap

# List Database
data = pd.read_csv('database.csv') # Jumlah data base
db = data['Identitas'] # Kolom Identitas berisikan ID
dba = db[-1:]
ID = db.tolist() # Menghilangkan name dan type
dbs = dba.to_string(index=False)

# Jendela satu

class Thread(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)

    def run(self):
        global td
        global th
        global tkn
        global tkr
        global waktud
        global rrdd
        global waktuh
        global rrhh
        global waktukn
        global rrknn
        global waktukr
        global rrkrr
        global rrd
        global rrh
        global rrkn
        global rrkr
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Masukan wajah
        cap = cv2.VideoCapture(0)  
        while True:
            ret, img = cap.read() # Baca img
            #print(img)
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # ubah img_matrix jd gray_matrix
                faces = face_cascade.detectMultiScale(gray,minSize=(170, 170),maxSize=(190,190)) # ada wajah pada jendela gray 
                for (x, y , w ,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 1) # Ganbar ROI di faces [x,y,w,h]
                    
                    # Dahi
                    cv2.rectangle(img, (x+30,y+10), (x+w-30, y+h-130), (255, 255 , 0), 1)
                    colorsLd = gray[y+10:y+h-130,x+30:x+w-30]
                    Ld.append(colorsLd)
                    #print(Ld)
                    nld = [i * M for i in Ld] # Perkalian gradien dengan Luminansi dahi
                    Ld.clear()
                    temd = [j + C for j in nld] #TDahi
                     #print(temd)
                    rrd = np.mean(temd) #TDahiRerata
                    #print(rrd)
                    
                    # Hidung
                    cv2.rectangle(img, (x+80,y+80), (x+w-80, y+h-70), (0, 255 , 255), 1)
                    colorsLh = gray[y+80:y+h-70,x+80:x+w-80]
                    Lh.append(colorsLh)
                    #print(Lh)
                    nlh = [i * M for i in Lh]# Perkalian gradien dengan Luminansi hidung
                    Lh.clear()
                    temh = [j + C for j in nlh] #THidung 
                    #print(temh)
                    rrh = np.mean(temh) #THidungRerata
                    #print(rrh)
                    
                    # Pipi Kanan
                    cv2.rectangle(img,(x+w-80,y+100), (x+w-30, y+h-50), (0, 140 , 255), 1)
                    colorsLkn = gray[y+100:y+h-50,x+w-80:x+w-30]
                    Lkn.append(colorsLkn)
                    #print(Lkn)
                    nlkn = [i * M for i in Lkn]# Perkalian gradien dengan Luminansi kanan
                    Lkn.clear()
                    temkn = [j + C for j in nlkn] #TPkanan
                    #print(temkn)
                    rrkn = np.mean(temkn) #TPKananRerata
                    #print(rrkn)
                    
                    # Pipi Kiri
                    cv2.rectangle(img,(x+30,y+100),(x+80,y+h-50), (0, 90 , 255), 1)
                    colorsLkr = gray[y+100:y+h-50,x+30:x+80]
                    Lkr.append(colorsLkr)
                    #print(Lkr)
                    nlkr = [i * M for i in Lkr]# Perkalian gradien dengan Luminansi kiri
                    Lkr.clear()
                    temkr = [j + C for j in nlkr] #TPKiri
                    #print(temkr)
                    rrkr = np.mean(temkr) #TPKiriRerata 
                    #print(rrkr)
                    
                h, w, ch = img.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(img.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = convertToQtFormat.scaled(720, 600, Qt.KeepAspectRatio) 
                self.changePixmap.emit(p)
            
class App(QtGui.QWidget):    
    
    def keluar(self):
        sys.exit(app.exec_()) 
        
    def simpan(self):
        img_df = pd.DataFrame(dframe ,columns = ['Identitas','Pasien','Jenis Kelamin','Operator','Usia Tahun','Usia Bulan','Waktu','Temperature Dahi','Temperature Hidung','Temperature Pipi Kanan','Temperature Pipi Kiri','Delta Dahi','Delta Hidung','Delta Pipi Kanan','Delta Pipi Kiri','Emosi'])
        img_df.to_csv('Olah_Suhu_Latih_%s.csv'%Namap,index = False)
        
    def __init__(self):
        super(App,self).__init__()
        self.title = 'Thermography Camera Interface - TCI'
        self.left = 480
        self.top = 40
        self.width = 640
        self.height = 480
        self.initUI()
                
    def setImage(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def initUI(self):
        global SuhuDahi
        global SuhuDahia
        global SuhuHidung
        global SuhuHidunga
        global SuhuPipiKanan
        global SuhuKanana
        global SuhuPipiKiri
        global SuhuKiria
        global JenEmosi
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1920, 1080)
        self.setStyleSheet("background-color: black;")
       
        # Label Kamera
        self.label = QtGui.QLabel(self)
        self.label.move(50, 140)
        self.label.resize(720,511)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        
        
        # Judul
        self.Judul = QtGui.QLabel(self)
        self.Judul.move(30, 30)
        self.Judul.setText('Thermography Camera Interface - TCI')
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(32)
        font.setItalic(True)
        font.setBold(True)
        font.setWeight(75)
        self.Judul.setFont(font)
        self.Judul.setStyleSheet("color: white;")
        
        # Dahi
        SuhuDahi = PlotWidget(self)
        SuhuDahi.setGeometry(QtCore.QRect(870, 170, 781, 221))
        SuhuDahi.setLabel('left',"Temperature (C)")
        SuhuDahi.setLabel('bottom',"Time (s)")
        SuhuDahi.showGrid(x=True,y=True)
        axisSuhuDahi = SuhuDahi.getAxis('bottom')
        axisSuhuDahi.setTickSpacing(20,5)
        
        self.TDahi = QtGui.QLabel(self)
        self.TDahi.move(870, 130)
        self.TDahi.setText('Suhu Dahi')
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        self.TDahi.setFont(font)
        self.TDahi.setStyleSheet("color: white;")
        self.SuhuDahit = QtGui.QLabel(self)
        self.SuhuDahit.move(1670, 180)
        self.SuhuDahit.setText("Suhu Dahi : ")
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        self.SuhuDahit.setFont(font)
        self.SuhuDahit.setStyleSheet("color: white;")
        SuhuDahia = QtGui.QLabel(self)
        SuhuDahia.move(1820, 180)
        SuhuDahia.resize(101,71)
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(10)
        font.setBold(True)
        SuhuDahia.setFont(font)
        SuhuDahia.setStyleSheet("color: white;")
        
        # Hidung
        SuhuHidung = PlotWidget(self)
        SuhuHidung.setGeometry(QtCore.QRect(870, 440, 781, 221))
        self.THidung = QtGui.QLabel(self)
        SuhuHidung.setLabel('left',"Temperature (C)")
        SuhuHidung.setLabel('bottom',"Time (s)")
        SuhuHidung.showGrid(x=True,y=True)
        axisSuhuHidung = SuhuHidung.getAxis('bottom')
        axisSuhuHidung.setTickSpacing(20,5)
        
        self.THidung.move(870, 400)
        self.THidung.setText('Suhu Hidung')
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setFamily("Cantarell")
        self.THidung.setFont(font)
        self.THidung.setStyleSheet("color: white;")
        self.SuhuHidungt = QtGui.QLabel(self)
        self.SuhuHidungt.move(1670, 240)
        self.SuhuHidungt.setText("Suhu Hidung : ")
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        self.SuhuHidungt.setFont(font)
        self.SuhuHidungt.setStyleSheet("color: white;")
        SuhuHidunga = QtGui.QLabel(self)
        SuhuHidunga.move(1820, 250)
        SuhuHidunga.resize(101,71)
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(10)
        font.setBold(True)
        SuhuHidunga.setFont(font)
        SuhuHidunga.setStyleSheet("color: white;")
        
        # Pipi Kanan
        SuhuPipiKanan = PlotWidget(self)
        SuhuPipiKanan.setGeometry(QtCore.QRect(870, 730, 781, 221))
        SuhuPipiKanan.setLabel('left',"Temperature (C)")
        SuhuPipiKanan.setLabel('bottom',"Time (s)")
        SuhuPipiKanan.showGrid(x=True,y=True)
        axisSuhuPipiKanan = SuhuPipiKanan.getAxis('bottom')
        axisSuhuPipiKanan.setTickSpacing(20,5)
        
        self.TKanan = QtGui.QLabel(self)
        self.TKanan.move(870, 680)
        self.TKanan.setText('Suhu Pipi Kanan')
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setFamily("Cantarell")
        self.TKanan.setFont(font)
        self.TKanan.setStyleSheet("color: white;")
        self.SuhuKanant = QtGui.QLabel(self)
        self.SuhuKanant.move(1670, 440)
        self.SuhuKanant.setText("Suhu Pipi Kanan : ")
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(10)
        font.setBold(True)
        self.SuhuKanant.setFont(font)
        self.SuhuKanant.setStyleSheet("color: white;")
        SuhuKanana = QtGui.QLabel(self)
        SuhuKanana.move(1820, 450)
        SuhuKanana.resize(101,71)
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(10)
        font.setBold(True)
        SuhuKanana.setFont(font)
        SuhuKanana.setStyleSheet("color: white;")
        
        # Pipi Kiri 
        SuhuPipiKiri = PlotWidget(self)
        SuhuPipiKiri.setGeometry(QtCore.QRect(50, 730, 781, 221))
        SuhuPipiKiri.setLabel('left',"Temperature (C)")
        SuhuPipiKiri.setLabel('bottom',"Time (s)")
        SuhuPipiKiri.showGrid(x=True,y=True)
        axisSuhuPipiKiri = SuhuPipiKiri.getAxis('bottom')
        axisSuhuPipiKiri.setTickSpacing(20,5)
        
        self.TKiri = QtGui.QLabel(self)
        self.TKiri.move(50, 685)
        self.TKiri.setText('Suhu Pipi Kiri')
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setFamily("Cantarell")
        self.TKiri.setFont(font)
        self.TKiri.setStyleSheet("color: white;")
        self.SuhuKirit = QtGui.QLabel(self)
        self.SuhuKirit.move(1670, 540)
        self.SuhuKirit.setText("Suhu Pipi Kiri : ")
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(10)
        font.setBold(True)
        self.SuhuKirit.setFont(font)
        self.SuhuKirit.setStyleSheet("color: white;")
        SuhuKiria = QtGui.QLabel(self)
        SuhuKiria.move(1820, 550)
        SuhuKiria.resize(101,71)
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(10)
        font.setBold(True)
        SuhuKiria.setFont(font)
        SuhuKiria.setStyleSheet("color: white;")
        
        # Jenis Emosi
        self.Emosi = QtGui.QLabel(self)
        self.Emosi.move(1280, 40)
        self.Emosi.setText('Jenis Emosi : ')
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.Emosi.setFont(font)
        self.Emosi.setStyleSheet("color: white;")
        
        # Emosi
        JenEmosi = QtGui.QLabel(self)
        JenEmosi.move(1500, 40)
        JenEmosi.resize(100,20)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        JenEmosi.setFont(font)
        JenEmosi.setStyleSheet("color: white;")
        
        # Menubar
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setStyleSheet("background-color: white;")
        self.actionFile = self.menubar.addMenu("File")
        self.start = self.actionFile.addAction("Start")
        self.save = self.actionFile.addAction("Save")
        self.save.triggered.connect(self.simpan)
        self.start.triggered.connect(self.mulai2)
        self.actionFile.addSeparator()
        self.quit = self.actionFile.addAction("Quit")
        self.quit.triggered.connect(self.keluar)
        self.actionHelp = self.menubar.addMenu("Help")
        # Init List
        self.dialogs = list()
        
    def mulai2(self):
        global dialog
        dialog = DataDiri()
        self.dialogs.append(dialog)
        dialog.show()
        
# Jendela Dua

class DataDiri(QtGui.QWidget):
    def __init__(self):
        super(DataDiri,self).__init__()
        self.title = 'Data Diri'
        self.left = 660
        self.top = 400
        self.width = 457
        self.height = 348
        self.initui()

        
    def mulai(self): 
        global Namap
        global tahun
        global bulan
        global jkelamin
        global Namao
        global Usiatr
        global Usiab
        
        incr = db[-1:] + 1
        ID.extend(incr.values)
        id_df = pd.DataFrame(ID,columns = ['Identitas'])
        id_df.to_csv('database.csv',index = False)
        
        # Menyimpan data-data 
        Namap = self.NamaPasien.text()
        tahun = self.Tanggal.date().year()
        bulan = self.Tanggal.date().month()
        jkelamin = str(self.PilihanKelamin.currentText())
        Namao = self.NamaOperatorr.text()
        Tahuns = datetime.now().year
        Bulans = datetime.now().month
        Usiat = Tahuns-tahun
        Usiab = Bulans-bulan
        if Usiab < 0:
            Usiatr = Usiat - 1
        else:
            Usiatr = Usiat
            
        lengkap.append([ID[-1:],Namap,jkelamin,Namao,Usiatr,abs(Usiab)])
        
        with open('dbase.csv', "a",newline='') as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(lengkap[0])
        
        self.close()
        self.timerb.start(5000)
        self.timerb.timeout.connect(self.akuisisi)
        self.timer.start(60000)
        self.timer.timeout.connect(self.emosi)
        
    def akuisisi(self):
        global td
        global th
        global tkn
        global tkr
        global waktud
        global rrdd
        global waktuh
        global rrhh
        global waktukn
        global rrknn
        global waktukr
        global rrkrr
        global rrd
        global rrh
        global rrkn
        global rrkr
        global waktud5
        global rrdd5
        global waktuh5
        global rrhh5
        global waktukn5
        global rrknn5
        global waktukr5
        global rrkrr5
        global JenEmosi
        global DTd
        global DTh
        global DTkn
        global DTkr
        
        # List Waktu
        td += 5
        th += 5
        tkn += 5
        tkr += 5
        # Dahi
        if rrd >= 26 and rrd<=39: 
            waktud.append(td)
            rrdd.append(rrd) #TDahiReratatn
            rrdb.append(rrdd[-14:])
            rrdbm = np.mean(rrdb[0]) #TDahiReratat60
            # print(rrdbm)
            DTd = rrd -rrdbm   #DeltaTDahiRerata
            #print(DTd)
            DelTd.append(DTd)
            
        waktud5 = waktud[-50:]
        rrdd5 = rrdd[-50:]

        SuhuDahi.plot(waktud5,rrdd5,clear = True,symbol='o')
        SuhuDahia.setText("%.1f"%rrd)

        # Hidung
        if rrh >= 26 and rrh<=39:
            waktuh.append(th)
            rrhh.append(rrh) #THidungReratatn
            rrhb.append(rrhh[-14:]) 
            rrhbm = np.mean(rrhb[0]) #THidungReratat60
            #print(rrhbm)
            DTh = rrh - rrhbm #DeltaTHidungRerata
            #print(DTh)
            DelTh.append(DTh)
        
        waktuh5 = waktuh[-50:]
        rrhh5 = rrhh[-50:]
        
        SuhuHidung.clear()
        SuhuHidung.plot(waktuh5,rrhh5, clear = True,symbol='o')
        SuhuHidunga.setText("%.1f"%rrh)
        
        # Pipi Kanan
        if rrkn >= 26 and rrkn<=39:
            waktukn.append(tkn)
            rrknn.append(rrkn) #THidungReratatn
            rrknb.append(rrknn[-14:]) 
            rrknbm = np.mean(rrknb[0]) #THidungReratat60
            #print(rrhbm)
            DTkn = rrkn - rrknbm #DeltaTHidungRerata
            #print(DTh)
            DelTkn.append(DTkn)
        
        waktukn5 = waktukn[-50:]
        rrknn5 = rrknn[-50:]
        
        SuhuPipiKanan.clear()    
        SuhuPipiKanan.plot(waktukn5,rrknn5,clear = True,symbol='o')
        SuhuKanana.setText("%.1f"%rrkn)
        
        # Pipi Kiri
        if rrkr >= 26 and rrkr<=39:
            waktukr.append(tkr)
            rrkrr.append(rrkr) #TPKiriReratatn
            rrkrb.append(rrkrr[-14:])
            rrkrbm = np.mean(rrkrb[0]) #TPKiriReratat60
            #print(rrkrbm)
            DTkr = rrkr - rrkrbm #DeltaTPKiriRerata
            #print(DTkr)
            DelTkr.append(DTkr)
        
        waktukr5 = waktukr[-50:]
        rrkrr5 = rrkrr[-50:]
        
        SuhuPipiKiri.clear()
        SuhuPipiKiri.plot(waktukr5,rrkrr5,clear=True,symbol='o')
        SuhuKiria.setText("%.1f"%rrkr)
        
        
        
        
    def emosi (self):
        global Emosii
        # Klasifikasi 
        # Definisi dari variabel yang akan digunakan
        # Variable Masukan
        SDahi = ctrl.Antecedent(np.arange(-0.5,0.5,0.01),'SDahi')
        SHidung = ctrl.Antecedent(np.arange(-0.5,0.5,0.01),'SHidung')
        SKanan = ctrl.Antecedent(np.arange(-0.5,0.5,0.01),'SKanan')
        SKiri = ctrl.Antecedent(np.arange(-0.5,0.5,0.001),'SKiri')
        HEmosi = ctrl.Consequent(np.arange(0,1.4,0.1),'HEmosi')
        
        # Membership Masukan
        SDahi['Low'] = fuzz.trimf(SDahi.universe,[-0.5,-0.5,-0.05])
        SDahi['Medl'] = fuzz.trimf(SDahi.universe,[-0.05,-0.025,-0.0125])
        SDahi['Med'] = fuzz.trimf(SDahi.universe,[-0.0125,0,0.0125])
        SDahi['Medh'] = fuzz.trimf(SDahi.universe,[0.0125,0.025,0.05])
        SDahi['Hi'] = fuzz.trimf(SDahi.universe,[0.05,0.5,0.5])
        
        SHidung['Low'] = fuzz.trimf(SHidung.universe,[-0.5,-0.5,-0.05])
        SHidung['Medl'] = fuzz.trimf(SHidung.universe,[-0.05,-0.025,-0.0125])
        SHidung['Med'] = fuzz.trimf(SHidung.universe,[-0.0125,0,0.0125])
        SHidung['Medh'] = fuzz.trimf(SHidung.universe,[0.0125,0.025,0.05])
        SHidung['Hi'] = fuzz.trimf(SHidung.universe,[0.05,0.5,0.5])
        
        SKanan['Low'] = fuzz.trimf(SKanan.universe,[-0.5,-0.5,-0.05])
        SKanan['Medl'] = fuzz.trimf(SKanan.universe,[-0.05,-0.025,-0.0125])
        SKanan['Med'] = fuzz.trimf(SKanan.universe,[-0.0125,0,0.0125])
        SKanan['Medh'] = fuzz.trimf(SKanan.universe,[0.0125,0.025,0.05])
        SKanan['Hi'] = fuzz.trimf(SKanan.universe,[0.05,0.5,0.5])
        
        SKiri['Low'] = fuzz.trimf(SKiri.universe,[-0.5,-0.5,-0.05])
        SKiri['Medl'] = fuzz.trimf(SKiri.universe,[-0.05,-0.025,-0.0125])
        SKiri['Med'] = fuzz.trimf(SKiri.universe,[-0.0125,0,0.0125])
        SKiri['Medh'] = fuzz.trimf(SKiri.universe,[0.0125,0.025,0.05])
        SKiri['Hi'] = fuzz.trimf(SKiri.universe,[0.05,0.5,0.5])
        
        # Membership Keluaran
        HEmosi['Non'] = fuzz.trimf(HEmosi.universe,[0,0.125,0.25])
        HEmosi['Senang'] = fuzz.trimf(HEmosi.universe,[0.25,0.375,0.5])
        HEmosi['Takut'] = fuzz.trimf(HEmosi.universe,[0.5,0.625,0.75])
        HEmosi['Sedih'] = fuzz.trimf(HEmosi.universe,[0.75,0.875,1])
        HEmosi['Tenang'] = fuzz.trimf(HEmosi.universe,[1,1.125,1.25])
        
        # Rule Base
        # Aturan Senang
        AturanSenang1 = ctrl.Rule(SDahi['Low'] & SHidung['Hi'],HEmosi['Senang'])
        AturanSenang2 = ctrl.Rule(SDahi['Medl'] & SHidung['Hi'],HEmosi['Senang'])
        AturanSenang3 = ctrl.Rule(SDahi['Hi'] & SHidung['Hi'],HEmosi['Senang'])
        AturanSenang4 = ctrl.Rule(SDahi['Medh'] & SHidung['Hi'],HEmosi['Senang'])
        AturanSenang5 = ctrl.Rule(SHidung['Hi'] & SKanan['Low'],HEmosi['Senang'])
        AturanSenang6 = ctrl.Rule(SHidung['Hi'] & SKiri['Low'],HEmosi['Senang'])
        AturanSenang7 = ctrl.Rule(SHidung['Hi'] & SKanan['Medl'],HEmosi['Senang'])
        AturanSenang8 = ctrl.Rule(SHidung['Hi'] & SKiri['Medl'],HEmosi['Senang'])
        AturanSenang9 = ctrl.Rule(SHidung['Hi'] & SKanan['Low'] & SKiri['Low'],HEmosi['Senang'])
        AturanSenang10 = ctrl.Rule(SHidung['Med']& SKanan['Low'] & SKiri['Low'],HEmosi['Senang'])
        AturanSenang11 = ctrl.Rule(SDahi['Low'] & SHidung['Med'],HEmosi['Senang'])
        AturanSenang12 = ctrl.Rule(SDahi['Low'] & SHidung['Medh'],HEmosi['Senang'])
        AturanSenang13 = ctrl.Rule(SDahi['Low'] & SHidung['Medh'],HEmosi['Senang'])
        AturanSenang14 = ctrl.Rule(SDahi['Med'] & SHidung['Hi'],HEmosi['Senang'])
        
        # Aturan Takut
        AturanTakut1 = ctrl.Rule(SHidung['Low'] & SKanan['Low'] & SKiri['Low'],HEmosi['Takut'])
        AturanTakut2 = ctrl.Rule(SHidung['Low'] & SKanan['Low'] & SKiri['Medl'],HEmosi['Takut'])
        AturanTakut3 = ctrl.Rule(SHidung['Low'] & SKanan['Medl'] & SKiri['Medl'],HEmosi['Takut'])
        AturanTakut4 = ctrl.Rule(SHidung['Medl'] & SKanan['Low'] & SKiri['Medl'],HEmosi['Takut'])
        AturanTakut5 = ctrl.Rule(SHidung['Medl'] & SKanan['Low'] & SKiri['Low'],HEmosi['Takut'])
        AturanTakut6 = ctrl.Rule(SDahi['Low'] & SKanan['Medl'] & SKiri['Low'],HEmosi['Takut'])
        AturanTakut7 = ctrl.Rule(SKanan['Medl'] & SKiri['Medl'],HEmosi['Takut'])
        AturanTakut8 = ctrl.Rule(SKanan['Med'] & SKiri['Medl'],HEmosi['Takut'])
        
        # Aturan Sedih
        AturanSedih1 = ctrl.Rule(SDahi['Low'] & SHidung['Low'] & SKanan['Hi'] & SKiri['Low'],HEmosi['Sedih'])
        AturanSedih2 = ctrl.Rule(SKanan['Medh'] & SKiri['Hi'],HEmosi['Sedih'])
        AturanSedih3 = ctrl.Rule(SDahi['Low'] & SHidung['Low'] & SKiri['Hi'] ,HEmosi['Sedih'])
        AturanSedih4 = ctrl.Rule(SKanan['Hi'] & SKiri['Hi'],HEmosi['Sedih'])
        AturanSedih5 = ctrl.Rule(SKanan['Hi'] & SKiri['Medh'] ,HEmosi['Sedih'])
        AturanSedih6 = ctrl.Rule(SHidung['Medl'] & SKanan['Hi'],HEmosi['Sedih'])
        AturanSedih7 = ctrl.Rule(SDahi['Medl'] & SKanan['Medh'],HEmosi['Sedih'])
        
        # Aturan Tenang
        AturanTenang1 = ctrl.Rule(SDahi['Medl'] & SHidung['Low'],HEmosi['Tenang'])
        AturanTenang2 = ctrl.Rule(SDahi['Low'] & SHidung['Low'] & SKiri['Medl'],HEmosi['Tenang'])
        AturanTenang3 = ctrl.Rule(SDahi['Low'] & SHidung['Low'] & SKanan['Low'] & SKiri['Low'],HEmosi['Tenang'])
        AturanTenang4 = ctrl.Rule(SDahi['Low'] & SHidung['Medl']  & SKiri['Low'],HEmosi['Tenang'])
        AturanTenang5 = ctrl.Rule(SDahi['Low'] & SHidung['Medl']  & SKiri['Med'],HEmosi['Tenang'])
        AturanTenang6 = ctrl.Rule(SDahi['Hi'] & SKanan['Low'] & SKiri['Low'],HEmosi['Tenang'])
        AturanTenang7 = ctrl.Rule(SDahi['Medl'] & SKanan['Low'] & SKiri['Low'],HEmosi['Tenang'])
        AturanTenang8 = ctrl.Rule(SDahi['Low'] & SHidung['Med'] & SKiri['Medl'],HEmosi['Tenang'])
        
        # Controller
        Emosi_ctrl = ctrl.ControlSystem([AturanSenang1,AturanSenang2,AturanSenang3,AturanSenang4,AturanSenang5
                                         ,AturanSenang6,AturanSenang7,AturanSenang8,AturanSenang9,AturanSenang10
                                         ,AturanSenang11,AturanSenang12,AturanSenang13,AturanSenang14
                                         ,AturanTenang1,AturanTenang2,AturanTenang3,AturanTenang4,AturanTenang5
                                         ,AturanTakut1,AturanTakut2,AturanTakut3,AturanTakut4,AturanTakut5
                                         ,AturanTakut6,AturanTakut7,AturanTakut8
                                         ,AturanSedih1,AturanSedih2,AturanSedih3,AturanSedih4,AturanSedih5
                                         ,AturanSedih6,AturanSedih7
                                         ,AturanTenang6,AturanTenang7,AturanTenang8
                                         ])
        JEmosi = ctrl.ControlSystemSimulation(Emosi_ctrl)
        JEmosi.input['SDahi'] = DTd
        JEmosi.input['SHidung'] = DTh
        JEmosi.input['SKanan'] = DTkn
        JEmosi.input['SKiri'] = DTkr
        
        JEmosi.compute()
        Keluaran = JEmosi.output['HEmosi']
        Emosii = ' '
        
        # PrintHasil
        if Keluaran >= 0 and Keluaran <= 0.25:
            Emosii = 'Non-Identified'
            JenEmosi.setText('Non-Identified')
        if Keluaran >= 0.25 and Keluaran <= 0.5:
            Emosii = 'Senang'
            JenEmosi.setText('Senang')
        if Keluaran >= 0.5 and Keluaran <= 0.75:
            Emosii = 'Takut'
            JenEmosi.setText('Takut')
        if Keluaran >= 0.75 and Keluaran <= 1:
            JenEmosi.setText('Sedih')
        if Keluaran >= 1 and Keluaran <= 1.25:
            Emosii = 'Tenang'
            JenEmosi.setText('Tenang')
        # Opsi penyimpanan
        dframe.append([ID[-1:],Namap,jkelamin,Namao,Usiatr,abs(Usiab),td,rrd,rrh,rrkn,rrkr,DTd,DTh,DTkn,DTkr,Emosii])
        
        
    def initui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(457, 348)
        self.setStyleSheet("background-color: black;")
        
        # Tombol Mulai
        self.TombolMulai = QtGui.QPushButton(self)
        self.TombolMulai.move(180,260)
        self.TombolMulai.resize(141,61)
        self.TombolMulai.setText('Mulai')
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(14)
        font.setBold(True)
        self.TombolMulai.setFont(font)
        self.TombolMulai.setStyleSheet("color: black;background-color:white")
        self.TombolMulai.clicked.connect(self.mulai)
        
        # Nama
        self.Nama = QtGui.QLabel(self)
        self.Nama.move(40, 80)
        self.Nama.resize(111, 20)
        self.Nama.setText('Nama Pasien')
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Nama.setFont(font)
        self.Nama.setStyleSheet("color: white;")
        self.NamaPasien = QtGui.QLineEdit(self)
        self.NamaPasien.move(170, 80)
        self.NamaPasien.resize(221, 24)
        self.NamaPasien.setStyleSheet("background-color: white; color:black")
        
        # ID
        self.ID = QtGui.QLabel(self)
        self.ID.move(130, 50)
        self.ID.resize(51, 16)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.ID.setFont(font)
        self.ID.setText("ID")
        self.ID.setStyleSheet("color: white;")
        self.IDisian = QtGui.QLabel(self)
        self.IDisian.move(170, 50) 
        self.IDisian.resize(101,20)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.IDisian.setFont(font)
        self.IDisian.setText(dbs)
        self.IDisian.setStyleSheet("color: white;")
        
        # Tanggal Lahir
        self.TanggalLahir = QtGui.QLabel(self)
        self.TanggalLahir.move(40, 120)
        self.TanggalLahir.resize(151, 31)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.TanggalLahir.setFont(font)
        self.TanggalLahir.setText("Tanggal Lahir")
        self.TanggalLahir.setStyleSheet("color: white;")
        self.Tanggal = QtGui.QDateEdit(self)
        self.Tanggal.move(170, 120)
        self.Tanggal.resize(110, 25)
        self.Tanggal.setStyleSheet("background-color: white; color:black")
        
        # Jenis Kelamin
        self.JenisKelamin = QtGui.QLabel(self)
        self.JenisKelamin.move(40, 160)
        self.JenisKelamin.resize(121, 31)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.JenisKelamin.setFont(font)
        self.JenisKelamin.setText("Jenis Kelamin")
        self.JenisKelamin.setStyleSheet("color: white;")
        self.PilihanKelamin = QtGui.QComboBox(self)
        self.PilihanKelamin.addItem("")
        self.PilihanKelamin.addItem("")
        self.PilihanKelamin.setItemText(0,"Laki-Laki")
        self.PilihanKelamin.setItemText(1,"Perempuan")
        self.PilihanKelamin.move(170, 160)
        self.PilihanKelamin.resize(101, 24)
        self.PilihanKelamin.setStyleSheet("background-color: white; color:black")
        
        # Operator
        self.NamaOperator = QtGui.QLabel(self)
        self.NamaOperator.move(20, 200) 
        self.NamaOperator.resize(141, 20)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NamaOperator.setFont(font)
        self.NamaOperator.setText("Nama Operator")
        self.NamaOperator.setStyleSheet("color: white;")
        self.NamaOperatorr = QtGui.QLineEdit(self)
        self.NamaOperatorr.move(170, 200) 
        self.NamaOperatorr.resize(221, 24)
        self.NamaOperatorr.setStyleSheet("background-color: white; color:black")
        
        # Timer
        # Timer Update
        self.timer = QtCore.QTimer()
        # Timer Beneran
        self.timerb = QtCore.QTimer()
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())   