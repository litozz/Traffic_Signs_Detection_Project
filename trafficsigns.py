#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from time import time



"""
La funcion loadImage carga una imagen en color o blanco y negro.
el parametro color debe valer: 0->blanco/negro, distinto de 0 -> 
"""
def loadImage(path,color):
	if(color=="COLOR"):
		im = cv2.imread(path,cv2.IMREAD_COLOR)
		return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
	elif(color=="GRAYSCALE"):
		im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		return im
	else:
		raise ValueError, "LoadImage color values must be COLOR or GRAYSCALE"


"""
La funcion paintImage pinta la imagen por pantalla
"""
def paintImage(image,windowtitle="",imagetitle="",axis=False):
	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
	plt.imshow(image),plt.title(imagetitle)
	if(not axis):
		plt.xticks([]),plt.yticks([])
	plt.show()



def convert3Channels(img):
	if(len(img.shape)<3):
		return cv2.merge([img,img,img])

"""
La funcion paintMatrixImages pinta un conjunto de imagenes en un lienzo
"""
def paintMatrixImages(imagematrix,imagetitles,windowtitle="",axis=False):
	nrow=len(imagematrix)
	ncol=len(imagematrix[0])

	prefix=int(str(nrow)+str(ncol))

	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
	
	for i in xrange(nrow):
		for j in xrange(len(imagematrix[i])):
			plt.subplot(int(str(prefix)+str(1+(i*ncol+j))))
			plt.imshow(imagematrix[i][j])
			plt.title(imagetitles[i][j])

			if(not axis):
				plt.xticks([]),plt.yticks([])

	plt.show()






"""
Lee los contornos de las figuras geometricas.
"""
def leerContornos():
	CV_RETR_EXTERNAL=0
	CV_CHAIN_APPROX_TC89_L1=3
	imgs=[loadImage("./dataset/contornos/"+str(i)+".png","GRAYSCALE") for i in xrange(3)] #4 aniade el cuadrado
	_contoursTest=[]
	
	for im in imgs:
		output = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,2)
		im2, ctest, hierarchy=cv2.findContours(output, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1)
		_contoursTest.append(ctest)

		
		#draw=np.zeros((output.shape[0],output.shape[1],3))
		#cv2.drawContours(draw, ctest, 0, (128,128,128),3)
		#paintImage(draw)

	return _contoursTest


"""
Lee los contornos de una imagen en concreto
"""
def leerContornosImagen(img):
	CV_RETR_EXTERNAL=0
	CV_CHAIN_APPROX_TC89_L1=3
	output = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	im2, ctest, hierarchy=cv2.findContours(output, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1)

	# for i,cont in enumerate(ctest):
	# 	draw=np.zeros((output.shape[0],output.shape[1],3))
	# 	print(draw)
	# 	cv2.drawContours(draw, ctest, i, (128,128,128),1)
	# 	paintImage(draw)

	return ctest

	
	

"""
Funcion que a partir de un conjunto de contornos de una imagen
obtinene la figura (triangulo,circulo,octogono o cuadrado) que más
se le parece al contorno dado, en caso contrario se devuelve figura
no encontrada
"""
def buscarForma(contornos_figuras, imagen, umbral):
	contornos_imagen=leerContornosImagen(imagen)

	best_dis=999999.9
	best_fig=-1
	best_cont=None
	for cont_img in contornos_imagen:
		#approx = cv2.approxPolyDP(cont_img,0.01,True)
		if(len(cont_img)>10):
			
			#draw=np.zeros((imagen_bn.shape[0],imagen_bn.shape[1],3))
			#cv2.drawContours(draw, [cont_img], 0, (128,128,128),1)
			#paintImage(draw)
			
			for i,cont in enumerate(contornos_figuras):
				#approx_figura = cv2.approxPolyDP(cont[0],0.01,True)	

				dis=cv2.matchShapes(cont_img,cont[0], 1, 0.0)
				#dis=cv2.matchShapes(approx,approx_figura, 1, 0.0)
				if(best_dis>dis):
					best_dis=dis
					best_fig=i
					best_cont=cont_img


	if(best_dis>umbral):
		best_fig=-1	

	return best_cont, best_fig, best_dis



# def buscarColorPOCO_EFICIENTE(imagen_color):
# 	r,g,b=cv2.split(imagen_color)
# 	nbins=17

# 	bins_r=np.zeros((nbins))
# 	bins_g=np.zeros((nbins))
# 	bins_b=np.zeros((nbins))

# 	for fila_r,fila_g,fila_b in zip(r,g,b):
# 		for pix_r,pix_g,pix_b in zip(fila_r,fila_g,fila_b):
# 			pos=(nbins-1)/255.0
# 			bins_r[int(pos*pix_r)]+=1
# 			bins_g[int(pos*pix_g)]+=1
# 			bins_b[int(pos*pix_b)]+=1

# 	return np.argmax(bins_r),np.argmax(bins_g),np.argmax(bins_b)

"""
Funcion para obtener los bins R,G,B
"""
def buscarColor(imagen_color):
	r,g,b=cv2.split(imagen_color)
	nbins=17

	perc=(nbins-1)/255.0
	r=np.uint8(perc*r)
	g=np.uint8(perc*g)
	b=np.uint8(perc*b)
	
	bins_r=[len(r[r==i]) for i in xrange(nbins)]
	bins_g=[len(g[g==i]) for i in xrange(nbins)]
	bins_b=[len(b[b==i]) for i in xrange(nbins)]

	return np.argmax(bins_r),np.argmax(bins_g),np.argmax(bins_b)


"""
Funcion para obtener el label de la figura
"""
def getLabel(best_fig):
	if(best_fig!=-1):
		return 1
	else:
		return -1

# def toStringFigure(figure):
# 	if(figure==0):
# 		return "triangulo"
# 	elif (figure==1):
# 		return "circulo"
# 	elif (figure==2):
# 		return "octogono"
# 	elif (figure==3):
# 		return "cuadrado"
# 	return "nada"

"""
Funcion para convertir la figura encontrada de entero a cadena
"""
def toStringFigure(figure):
	#if(figure==0):
	#	return "triangulo"
	#elif (figure==1):
	#	return "circulo"
	#elif (figure==2):
	#	return "octogono"
	#elif (figure==3):
	#	return "cuadrado"
	if(figure==0 or figure==1 or figure==2 or figure==3):
		return "forma"
	return "nada"



"""
Funcion para obtener el vector de caracteristicas (X1,X2,X3,X4) siendo:
X1 -> figura detectada (forma|nada)
X2 -> bin correspondiente con el color Rojo
X3 -> bin correspondiente con el color Verde
X4 -> bin correspondiente con el color Azul
"""
def getVectorCaracteristicas(contornos_figuras,imagen_color,withLabel=True):
	imagen_bn=cv2.cvtColor(imagen_color,cv2.COLOR_BGR2GRAY)
	best_cont, best_fig, best_dis = buscarForma(contornos_figuras,imagen_bn,0.05)
	if(best_fig!=-1):
		x,y,w,h = cv2.boundingRect(best_cont)
		#cuadro=cv2.rectangle(imagen_color,(x,y),(x+w,y+h),(0,255,0),1)
		recortada=imagen_color[y:y+h,x:x+w]
		r,g,b=buscarColor(recortada)
		#paintImage(recortada)
	else:
		r,g,b=buscarColor(imagen_color)
	#print("Mejor figura: {}, mejor distancia: {}".format(best_fig,best_dis))
	#print("Color: {}".format([r,g,b]))
	#print("Vector caracteristicas: {}".format([toStringFigure(best_fig),r,g,b,getLabel(best_fig)]))
	if(not withLabel):
		vc=[toStringFigure(best_fig),r,g,b]
		newvc=""
		for elemento in vc:
			newvc=newvc+str(elemento)+";"
		return newvc[:-1]
	else:
		vc=[toStringFigure(best_fig),r,g,b,getLabel(best_fig)]
		newvc=""
		for elemento in vc:
			newvc=newvc+str(elemento)+";"
		return newvc[:-1]
	#paintImage(np.uint8([[[r,g,b]]]))

	#paintMatrixImages([[contornos_figuras[0],contornos_figuras[1],contornos_figuras[2]]],[["","",""]],"")



"""
Funcion para recorrer una imagen con una ventana
"""
def pasarVentana(imagen,tam_v,incremento,contornos_figuras):
	coordenadas=[]
	
	y=0
	x=0

	alto=imagen.shape[0] #Numero de filas
	ancho=imagen.shape[1] #Numero de columnas

	cont=1
	while(y+tam_v[0] <= alto and x+tam_v[1] <= ancho):
		imagen_a_analizar=imagen[y:y+tam_v[0]   , x:x+tam_v[1]  ]
		vc=getVectorCaracteristicas(contornos_figuras,imagen_a_analizar)
				
		if(vc.split(";")[0]!="nada"):
			coordenadas.append([(x,y),(x,y+tam_v[1]),(x+tam_v[0],y),(x+tam_v[0],y+tam_v[1])])
			#coordenadas.append([(fil,col),(fil,col+tam_v[1]),(fil+tam_v[0],col),(fil+tam_v[0],col+tam_v[1])])	
			#print("Vector de caracteristicas: {} ---------> Coordenadas: {}".format(vc,[(x,y),(x,y+tam_v[1]),(x+tam_v[0],y),(x+tam_v[0],y+tam_v[1])]))
		
		x+=incremento
		if(x+tam_v[1]>=ancho+1):
			#print("AUMENTO FILA")
			x=0
			y+=incremento
		
		
	return coordenadas





"""
Funcion para recorrer una imagen con una ventana, ademas incluye el 
algoritmo de machine learning que a partir del vector de caracteristicas
obtenido de la ventana predice si hay o no senial
"""
def pasarVentanaClasificador(imagen,tam_v,incremento,contornos_figuras,clf):
	coordenadas=[]
	
	y=0
	x=0

	alto=imagen.shape[0] #Numero de filas
	ancho=imagen.shape[1] #Numero de columnas

	cont=1
	while(y+tam_v[0] <= alto and x+tam_v[1] <= ancho):
		imagen_a_analizar=imagen[y:y+tam_v[0]   , x:x+tam_v[1]  ]
		
		vc=getVectorCaracteristicas(contornos,imagen_a_analizar,withLabel=False)
		ds=np.array([vc.split(";")])	
		ds=ds.T
		le=preprocessing.LabelEncoder()
		le.fit(["forma","nada"])
		ds[0]=list(le.transform(ds[0]))
		ds=ds.T
		prediccion=clf.predict(ds)

		#cv2.rectangle(imagen, (x,y), (x+tam_v[0],y+tam_v[1]), (255,0,0), 2)
		#print("{} ------> {}".format(vc,prediccion))
		#paintImage(imagen)
		

		if(prediccion==["1"]):
			coordenadas.append([(x,y),(x,y+tam_v[1]),(x+tam_v[0],y),(x+tam_v[0],y+tam_v[1])])
			#print("Area potencialmente interesante: {}".format([(x,y),(x,y+tam_v[1]),(x+tam_v[0],y),(x+tam_v[0],y+tam_v[1])]))
			#print(ds)
			if(ds[0][0]=="0"):
				ds[0][0]="forma"
			else:
				ds[0][0]="nada"
			newvc=""
			for elemento in ds[0]:
				newvc=newvc+str(elemento)+";"
			print(newvc[0:-1]+";-1")
		#else:
		#	print("Aqui no hay na: {}".format([(x,y),(x,y+tam_v[1]),(x+tam_v[0],y),(x+tam_v[0],y+tam_v[1])]))
			#coordenadas.append([(fil,col),(fil,col+tam_v[1]),(fil+tam_v[0],col),(fil+tam_v[0],col+tam_v[1])])	
			#print("Vector de caracteristicas: {} ---------> Coordenadas: {}".format(vc,[(x,y),(x,y+tam_v[1]),(x+tam_v[0],y),(x+tam_v[0],y+tam_v[1])]))
		
		x+=incremento
		if(x+tam_v[1]>=ancho+1):
			#print("AUMENTO FILA")
			x=0
			y+=incremento
		
		
	return coordenadas













"""
Funcion para leer el fichero dataset
"""
def leer_fichero(nombre_fichero):
    dataset=[]
    with open(nombre_fichero,"r") as f:
    	complete_file=f.read().split("\n")
    	dataset=np.array([reg.split(";") for reg in complete_file])
    return dataset



"""
Funcion para reducir una imagen manteniendo el aspect ratio
"""
def reduceImagen(imagen,nuevo_ancho,tamanio_ventana):
	if(nuevo_ancho<imagen.shape[1]):
		nuevo_alto=int((float(imagen.shape[0])/float(imagen.shape[1]))*nuevo_ancho)
		slicer_alto=int(float(imagen.shape[0])/nuevo_alto)
		slicer_ancho=int(float(imagen.shape[1])/nuevo_ancho)
		nueva_imagen=cv2.GaussianBlur(imagen[0:imagen.shape[0]:slicer_alto,0:imagen.shape[1]:slicer_ancho,::],tamanio_ventana,sigmaX=0,sigmaY=0)
		return slicer_alto,slicer_ancho,nueva_imagen
	else:
		return 1,1,imagen




if __name__=="__main__":

	dataset = leer_fichero("./dataset.txt")
	np.random.shuffle(dataset)

	dataset=dataset.T
	le=preprocessing.LabelEncoder()
	le.fit(["forma","nada"])
	dataset[0]=list(le.transform(dataset[0]))
	dataset=dataset.T

	numero_ejemplos=len(dataset)

	data=dataset[0:numero_ejemplos,0:4]
	labels=dataset[0:numero_ejemplos,4]

	clf=tree.DecisionTreeClassifier()
	clf=clf.fit(data,labels)

	
	contornos=leerContornos()
	#imagen_color=loadImage("./carretera4.jpg","COLOR")
	imagen_color=loadImage("./imagenes_nuevas/circular2/circular0019.jpg","COLOR")
	imagen_color_copia=imagen_color
	imagen_color=cv2.cvtColor(imagen_color,cv2.COLOR_RGB2HSV)
	print(imagen_color.shape)
	slicer_alto,slicer_ancho,imagen_color_2=reduceImagen(imagen_color,500,(3,3))
	print(imagen_color_2.shape)
	
	#paintMatrixImages([[imagen_color,imagen_color_2]],[["original","reducida"]],"")
	

	tiempo_inicial = time() 
	coords_r=pasarVentanaClasificador(imagen_color_2,(50,50),5,contornos,clf)
	coords_g=pasarVentanaClasificador(imagen_color_2,(75,75),10,contornos,clf)
	coords_b=pasarVentanaClasificador(imagen_color_2,(100,100),15,contornos,clf)

	for cr in coords_r:
	 	cv2.rectangle(imagen_color_copia, (slicer_alto*cr[0][0],slicer_ancho*cr[0][1]), (slicer_alto*cr[3][0],slicer_ancho*cr[3][1]), (255,0,0), 2)
	for cg in coords_g:
	  	cv2.rectangle(imagen_color_copia, (slicer_alto*cg[0][0],slicer_ancho*cg[0][1]), (slicer_alto*cg[3][0],slicer_ancho*cg[3][1]), (0,255,0), 2)
	for cb in coords_b:
	 	cv2.rectangle(imagen_color_copia, (slicer_alto*cb[0][0],slicer_ancho*cb[0][1]), (slicer_alto*cb[3][0],slicer_ancho*cb[3][1]), (0,0,255), 2)

	tiempo_final = time() 
	print("tiempo de procesamiento: {} segundos".format(tiempo_final-tiempo_inicial))
	paintImage(imagen_color_copia)