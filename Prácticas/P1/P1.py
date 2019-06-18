#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizaje Automático
Práctica 1

Alfonso García Martínez

Marzo de 2019
"""

import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
    1. Ejercicio sobre la búsqueda iterativa de óptimos
"""

# 1.Implementar el algoritmo de gradiente descendente
# Además de las iteraciones ejecutadas y el punto óptimo encontrado,
# devolveremos un vector de los puntos explorados y otro vector
# con los gradientes en esos puntos
def gradient_descent_array(gradient, starting_point, learning_rate=0.01,
                           max_iterations=100000, precision=10**(-20)):
    current_point = starting_point.copy()
    iterations = 0
    gradient_value = gradient(current_point)
    
    points = []
    gradients = []
    points.append(current_point.copy())
    gradients.append(gradient_value.copy())
    step_size = precision + 9999999.9999
    
    # Como criterios de parada, usamos un número máximo de iteraciones o
    # que la norma de la diferencia entre el punto actual y el de la iteración
    # anterior disminuya hasta un cierto valor
    while iterations < max_iterations and step_size > precision:
        previous_point = current_point.copy()
        current_point -= learning_rate*gradient_value
        iterations += 1
        gradient_value = gradient(current_point)
        step_size = np.linalg.norm(current_point-previous_point)
        # Añadimos el punto y gradiente actuales a las respectivas listas
        points.append(current_point.copy())
        gradients.append(gradient_value.copy())

    
    # Convertimos las listas de puntos y gradientes en arrays de numpy
    points_np = np.array(points)
    gradients_np = np.array(gradients)
    
    return current_point, iterations, points_np, gradients_np

# 2.Considerar la función E(u, v). Usar gradiente descendente
# para encontrar un mínimo de esta función, comenzando desde el punto 
# (u, v) = (1, 1) y usando una tasa de aprendizaje η = 0,01.


# Función E(u,v)
def E(u,v):
    return (u*u*math.exp(v) - 2*v*v*math.exp(-u))**2


# a) Calcula analíticamente y mostrar la expresión del gradiente de
#    la función E(u,v)

# Gradiente de E(u,v) calculado analíticamente 
# - recibe como parámetro un array de numpy con las coordenadas 2D de un punto
# - devuelve un array de numpy con el vector de gradiente en ese punto
def dfE(uv):
    u = uv[0]
    v = uv[1]
    du = 2*(u*u*math.exp(v)-2*v*v*math.exp(-u))*(2*math.exp(v)*u+2*v*v*math.exp(-u))
    dv = 2*(u*u*(math.exp(v))-2*v*v*math.exp(-u))*(u*u*math.exp(v)-4*math.exp(-u)*v)
    dE = np.array([du,dv]) 
    return dE



# Punto de inicio que usaremos para E(u,v)
spe = np.array([1.0,1.0])


# Ejecutamos sucesivamente el gradiente descendiente para ver con cuántas
# iteraciones y en qué punto se consigue que E(u,v) valga menos de 10^(-14)
w, it, pts, grs = gradient_descent_array(dfE, spe.copy(), 0.01, 20, 10**(-8))
max_it = it
print("Mímimo de E(u,v) encontrado: %.15f" % E(w[0],w[1]))

while E(w[0],w[1]) >= 10**(-14):
    w, it, pts, grs= gradient_descent_array(dfE, spe.copy(), 0.01, max_it, 10**(-8))
    max_it = it+1
    print("Para un máximo de %d iteraciones:" % max_it)
    print("Mímimo de E(u,v) encontrado: %.15f" % E(w[0],w[1]))
    print("-------------------------------------------------")

print("Iteraciones necesarias para alcanzar precisión de 10^-14: %d\n" % max_it)
print("Coordenadas (u,v) donde se encuentra el mínimo:")
print("(%.15f,%.15f)" % (w[0],w[1]))

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# 3.- Considerar ahora la función f(x,y)...

def f(x,y):
    return x*x + 2*y*y + 2*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

# Gradiente de f(x,y)
def df(xy):
    x = xy[0];
    y = xy[1];
    dx = 2*x + 4*math.pi*math.sin(2*math.pi*y)*math.cos(2*math.pi*x);
    dy = 4*y + 4*math.pi*math.sin(2*math.pi*x)*math.cos(2*math.pi*y);
    return np.array([dx,dy])


# a.

# Punto de inicio: (0.1, 0.1)
starting_point_f = np.array([0.1,0.1])

# Learning rate
eta = 0.01

op, it, pts, grs = gradient_descent_array(df, starting_point_f,
                                          learning_rate=eta,max_iterations=50)

print("Mínimo de f encontrado con learning rate de 0.01 tras 50 iteraciones: %f" % f(op[0],op[1]))
print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

# Imprimimos cómo evoluciona el valor de f(x,y)



# Vectorizamos la función f(x,y) para poder aplicarla directamente sobre 
# el grid de las componentes 'x' e 'y' para obtener las componentes 'z'
vectorized_f = np.vectorize(f)
x_points = pts[:,0]
y_points = pts[:,1]
f_points = vectorized_f(x_points, y_points)
iterations = np.arange(0,len(pts),1)

plt.clf()
plt.plot(iterations,f_points)
plt.xlabel("Iteración")
plt.ylabel("f(x,y)")
plt.title("Valores de f(x,y) explorados en Gradiente Descendiente")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# Imprimimos cómo desciende el gradiente

# Imprimimos la función f(x,y) en una gráfica 3D
a = np.arange(-0.3, 0.3, 0.02)
b = np.arange(-0.3, 0.3, 0.02)

# Creamos la cuadrícula con valores 'x' e 'y' para hacer la gráfica
x, y = np.meshgrid(a, b)

z = vectorized_f(x, y)
fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': '3d'})

ax.plot_surface(x, y, z, edgecolor='none', rstride=1,
                            cstride=1, cmap='ocean', alpha=0.4)


# Imprimimos todos los puntos que se han explorado en
# el proceso iterativo de gradiente descendiente
ax.scatter(x_points, y_points, f_points, c="Red", alpha=1)



# Imprimimos los gradientes de cada punto
# En realidad, pintaremos los opuestos de los gradientes 
# multiplicados por el learning rate para que se vean mejor
# las direcciones seguidas en el proceso de búsqueda
x_grads = -eta * grs[:,0]
y_grads = -eta * grs[:,1]
z_grads = np.zeros(len(grs[:,0])) # pondremos componente z nulo

ax.quiver(x_points, y_points, f_points, x_grads, y_grads, z_grads, color='Orange')
plt.title("Descenso por f(x,y) para un learning rate de 0,01")
plt.xlabel("x")
plt.ylabel("y")
ax.view_init(45, -70)
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

# Imprimimos también el descenso en 2D, pintando los contornos de f(x,y)
fig, ax = plt.subplots()
# Contorno de f(x,y)
ax.contour(x,y,z,cmap='ocean')
# Puntos explorados
ax.scatter(x_points, y_points, c="Red")
plt.title("Descenso por f(x,y) para un learning rate de 0,01 (en 2D)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# Reperimos el experimento cambiando el learning rate a 0.1
eta=0.1
op, it, pts, grs = gradient_descent_array(df, starting_point_f.copy(),
                                          learning_rate=eta,max_iterations=50)
print("Imagen en f del punto óptimo con learning rate de 0.1 tras 50 iteraciones: %f" % f(op[0],op[1]))

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

# Imprimimos cómo evoluciona el valor de f(x,y)


x_points = pts[:,0]
y_points = pts[:,1]
f_points = vectorized_f(x_points, y_points)
iterations = np.arange(0,len(pts),1)

plt.clf()
plt.plot(iterations,f_points)
plt.xlabel("Iteración")
plt.ylabel("f(x,y)")
plt.title("Valores de f(x,y) explorados en Gradiente Descendiente (learning rate 0,1)")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

# Imprimimos cómo desciende el gradiente de nuevo

# Imprimimos la función f(x,y) en una gráfica 3D
a = np.arange(-2.0, 2.0, 0.16)
b = np.arange(-2.0, 2.0, 0.16)

# Creamos la cuadrícula con valores 'x' e 'y' para hacer la gráfica
x, y = np.meshgrid(a, b)
z = vectorized_f(x, y)
fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': '3d'})

ax.plot_surface(x, y, z, edgecolor='none', rstride=1,
                            cstride=1, cmap='ocean', alpha=0.4)


# Imprimimos todos los puntos que se han explorado en
# el proceso iterativo de gradiente descendiente
x_points = pts[:,0]
y_points = pts[:,1]
z_points = vectorized_f(x_points, y_points)

ax.scatter(x_points, y_points, z_points, c="Red", alpha=1)


# Imprimimos los gradientes de cada punto
# En realidad, pintaremos los opuestos de los gradientes 
# multiplicados por el learning rate para que se aprecie mejor
# las direcciones seguidas en el proceso de búsqueda
x_grads = -eta * grs[:,0]
y_grads = -eta * grs[:,1]
z_grads = np.zeros(len(grs[:,0])) # pondremos componente z nulo

ax.quiver(x_points, y_points, z_points, x_grads, y_grads, z_grads, color='Orange')
plt.title("Descenso por f(x,y) para un learning rate de 0,1")
plt.xlabel("x")
plt.ylabel("y")
ax.view_init(53, -40)
plt.show()


print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# Imprimimos también el descenso en 2D, pintando los contornos de f(x,y)
fig, ax = plt.subplots()
# Contorno de f(x,y)
ax.contour(x,y,z,cmap='ocean')
# Puntos explorados
ax.scatter(x_points, y_points, c="Red")
plt.title("Descenso por f(x,y) para un learning rate de 0,1 (en 2D)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

# b.

# Lista con puntos de inicio
starting_points = [np.array([0.1,0.1]),np.array([1.0,1.0]),
                   np.array([-0.5,-0.5]),np.array([-1.0,-1.0])]

# Guardamos los valores mínimos y las coordenadas de sus respectivos puntos
minimums = []
points = []

# Para cada punto de inicio definido, ejecutamos el gradiente descendente
# para saber qué mínimo se alcanza y en qué punto
for starting_point in starting_points:
    p, it, pts, grs = gradient_descent_array(df, starting_point.copy(), 0.01, 10000, 10**(-8))
    points.append(p)
    minimums.append(f(p[0],p[1]))

# Una vez tenemos los resultados para distintos puntos de inicio,
# los imprimimos en una 'tabla' por pantalla

print("Mínimos de f(x,y) obtenidos a partir de distintos puntos de inicio:\n")

print("Punto de inicio\tValor mínimo\tCoordenadas de (x,y)")
print("---------------------------------------------------")

for i in range(0,len(starting_points),1):
    p = points[i]
    m = minimums[i]
    stp = starting_points[i]
    print("%s   \t%f  \t%s" % (np.array2string(stp,precision=2, separator=','), m,
                               np.array2string(p,precision=2, separator=',')))


print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()



"""
    2. Ejercicio regresión lineal
"""
# Antes de nada, hay que cargar el dataset de los ficheros

# Conjunto de test, separando las etiquetas (y) de las características (X)
features_test = np.load("datos/features_test.npy")
X_tst = features_test[:,1:len(features_test[0])]
y_tst = features_test[:,0]

# Conjunto de entrenamiento, separando las etiquetas (y) de las características (X)
features_train = np.load("datos/features_train.npy")
X_tra = features_train[:,1:len(features_train[0])]
y_tra = features_train[:,0]

# Hacemos también una pequeña transformación en los datos:
# convertimos las etiquetas '5' por '-1' para poder realizar el ajuste
np.place(y_tst, y_tst==5, [-1])
np.place(y_tra, y_tra==5, [-1])

# Borramos los arrays originales de los ficheros, pues ya no son necesarios
del features_train
del features_test




# 1. Estimar un modelo de regresión lineal a partir de
# los datos proporcionados de dichos números...



# Función para añadir una columna de 1's a los datos,
# para poder ajustar también el término independiente
def add_ones_column(data):
    ones = np.full((len(data),1),1.0)
    concatenated = np.concatenate((ones,data),axis=1)
    return concatenated



def generate_minibatches(data_x, data_y, minibatch_size):

    # Concatenamos los datos X con una columna con etiquetas y
    data_x_y = np.empty((len(data_y),len(data_x[0])+1))
    data_x_y[:,0:len(data_x[0])] = data_x.copy()
    data_x_y[:,-1] = np.transpose(data_y)
    # Desordenamos/barajamos los datos
    np.random.shuffle(data_x_y)
    
    # Lista con tuplas de arrays, cada tupla es un minibatch
    # con los valores x en un elemento de la tupla y
    # correspondientes etiquetas 'y' en el otro elemento de
    # la misma tupla
    minibatches = []
    
    # Sabiendo el tamaño de los minibatches, podemos
    # calcular el número de minibatches
    n_minibatches = len(data_y) // minibatch_size 
  
    minibatch_begin = 0
    minibatch_end = minibatch_size
    
    for i in range(0,n_minibatches + 1,1): 
        mini = data_x_y[minibatch_begin:minibatch_end]
        mini_x = mini[:,0:len(data_x[0])]
        mini_y = mini[:,-1]
        minibatches.append((mini_x,mini_y))
        minibatch_begin += minibatch_size
        minibatch_end  += minibatch_size
    
    return minibatches 


    
# Implementación del Gradiente Descendiente Estocástico
# con tamaño de minibatch ajustable
def sgd(gradient, error, starting_point, data_X, data_y, learning_rate=0.01, 
        max_iterations=3000, precision=10**(-20), minibatch_size=32, random_seed=None):
    current_point = starting_point.copy()
    iterations = 0
    
    # Fijamos semilla aleatoria en función del parámetro 'random_seed'
    if random_seed==None:
        np.random.seed()
    else:
        np.random.seed(random_seed)
    

    # Generar minibatches
    minibatches = generate_minibatches(data_X, data_y, minibatch_size)
    
    i = 0
    minibatch_X = minibatches[i][0]
    minibatch_y = minibatches[i][1]
    gradient_value = gradient_value = gradient(current_point,minibatch_X,minibatch_y)
    
    # Calcular error del punto actual
    err = error(current_point,data_X,data_y)
    
    while iterations < max_iterations and err > precision:
        current_point -= learning_rate*gradient_value
        iterations += 1
        gradient_value = gradient(current_point,minibatch_X,minibatch_y)
        err = error(current_point,data_X,data_y)
        # Siguiente minibatch
        i = (i+1)%len(minibatches)
        

    return current_point, iterations




# Implementación del método clásico de ajuste por la pseudo-inversa
def pseudo_inverse_fit(data_x,data_y):
    X = data_x.copy()
    y = data_y.copy()
    
    # pseudoinversa = (X^t X)^-1 X^T 
    pseudo = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))
    w = np.dot(pseudo,y)
    return w


# Error cuadrático dado un modelo lineal y un conjunto de datos etiquetado
def mean_squared_error(linear_model_weights, data_X, data_y):
    n = len(data_y)
        
    error_array = np.dot(data_X,np.transpose(linear_model_weights)) - np.transpose(data_y)

    sqrd_error = np.linalg.norm(error_array)
    sqrd_error = sqrd_error**2
    
    sqrd_error = sqrd_error/n*1.0
    
    
    
    return sqrd_error


# Gradiente del error cuadrático
# Se le pasa como parámetros el vector de pesos de nuestro modelo lineal,
# los datos y sus respectivas etiquetas (valores verdaderos)
def squared_error_gradient(linear_model_weights,data_X,data_y):
    n = len(data_y)
    
    error_array = np.dot(data_X,np.transpose(linear_model_weights)) - np.transpose(data_y)
    
    error_gradient = np.dot(np.transpose(data_X),error_array)
    error_gradient = 2.0*error_gradient/n
    
    return error_gradient


def linear(w,x):
    return (w[1]/w[2])*x+w[0]/w[2]


X_tra_ones = add_ones_column(X_tra)



# Vector de pesos inicial para la regresión lineal
# con valores iniciales aleatorios.
# Usamos semilla aleatoria para que al repetir
# siempre obtengamos los mismos valores.
np.random.seed(12345)
w = np.random.random(len(X_tra[0])+1)


# Errores de la regresión con gradiente descendiente estocástico
w_fitted_sgd, it = sgd(squared_error_gradient,mean_squared_error, w, X_tra_ones, y_tra, max_iterations=8000, random_seed=12345)
print("Modelo de regresión lineal ajustado con sgd en %d iteraciones:" % it)
print(w_fitted_sgd)

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

e_in_sgd = mean_squared_error(w_fitted_sgd,X_tra_ones, y_tra)
print("Error DENTRO de la muestra con sgd: %f" % e_in_sgd)

X_tst_ones = add_ones_column(X_tst)

e_out_sgd = mean_squared_error(w_fitted_sgd,X_tst_ones, y_tst)
print("Error FUERA de la muestra con sgd: %f" % e_out_sgd)
#gradient_descent_array(squared_error_gradient,w)

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# Pintamos el resultado obtenido
t = np.arange(0.0,0.6,0.02)
w = w_fitted_sgd

plt.clf()
plt.scatter(X_tra[y_tra==1,0],X_tra[y_tra==1,1],c="Blue",label="Dígito 1")
plt.scatter(X_tra[y_tra==-1,0],X_tra[y_tra==-1,1],c="Red",label="Dígito 5")
# Pintamos la resta a partir del plano de regresión a partir
# de la intersección del plano de regresión con el plano z=0
plt.plot(t,-(w[1]/w[2])*t -(w[0]/w[2]),c="Green",label="Ajuste con sgd (intersección w0 + x1w1 + x2w2 = 0)")
plt.xlabel("Intensidad promedio")
plt.ylabel("Simetría")
plt.legend()
plt.title("Regresión lineal con Gradiente Descendiente Estocástico (tamaño de minibatch: 32)")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


                     
                     
# Errores de la regresión con método de la pseudoinversa
w_fitted_pseudo = pseudo_inverse_fit(X_tra_ones, y_tra)
print("Modelo de regresión lineal ajustado con pseudoinversa:")
print(w_fitted_pseudo)






print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


e_in_pseudo = mean_squared_error(w_fitted_pseudo,X_tra_ones, y_tra)
print("Error DENTRO de la muestra con preudoinversa: %f" % e_in_pseudo)


e_out_sgd = mean_squared_error(w_fitted_pseudo,X_tst_ones, y_tst)
print("Error FUERA de la muestra con preudoinversa: %f" % e_out_sgd)


print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

w = w_fitted_pseudo

plt.clf()
plt.scatter(X_tra[y_tra==1,0],X_tra[y_tra==1,1],c="Blue",label="Dígito 1")
plt.scatter(X_tra[y_tra==-1,0],X_tra[y_tra==-1,1],c="Red",label="Dígito 5")
# Pintamos la resta a partir del plano de regresión a partir
# de la intersección del plano de regresión con el plano z=0
plt.plot(t,-(w[1]/w[2])*t -(w[0]/w[2]),c="Green",label="Ajuste con pseudoinversa (intersección w0 + x1w1 + x2w2 = 0)")
plt.xlabel("Intensidad promedio")
plt.ylabel("Simetría")
plt.legend()
plt.title("Regresión lineal con método de la pseudoinversa")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# 2. En este apartado exploramos como se transforman los errores E in y E out cuando au-
# mentamos la complejidad del modelo lineal usado...

# Generar muestra aleatoria de 'dim' dimensiones uniformemente distribuida
# en el cuadrado [-size,size]x[-size,size]
def simula_unif (N, dim, size):
    return np.random.uniform(-size,size,(N,dim))


# a. 

# Muestra de 1000 elementos en 2D en el cuadrado [-1,1]x[-1,1]
np.random.seed(12345)
unif_X_tra = simula_unif(1000,2,1.0)

# Pintamos la muestra con un scatter plot
plt.clf()
plt.scatter(unif_X_tra[:,0], unif_X_tra[:,1], c="Black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Muestra aleatoria uniforme de 1000 elementos")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

# b.

# Función para generar etiquetas para la muestra (-1 ó 1)
def f_label(x1,x2):
    f = (x1 - 0.2)**2 + x2**2 - 0.6
    return math.copysign(1,f)


# Generamos las etiquetas para la muestra,
# vectorizando la función previamente definida
f_label_vectorized = np.vectorize(f_label)
unif_y_tra = f_label_vectorized(unif_X_tra[:,0],unif_X_tra[:,1])


# Pintamos la muestra etiquetada
plt.clf()
plt.scatter(unif_X_tra[unif_y_tra==1,0], unif_X_tra[unif_y_tra==1,1], c="Blue", label='y = 1')
plt.scatter(unif_X_tra[unif_y_tra==-1,0], unif_X_tra[unif_y_tra==-1,1], c="Red", label='y = -1')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Muestra aleatoria uniforme de 1000 elementos etiquetados")
plt.legend()
plt.show()


print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# Introducimos ruido aleatorio, cambiando el signo de las etiquetas
# de un 10% de la muestra
random.seed(12345)
for i in range(0,len(unif_y_tra)//10,1):
    rand_i = random.randint(0,len(unif_y_tra)-1)
    unif_y_tra[rand_i] = -unif_y_tra[rand_i]

# Pintamos la muestra con ruido
plt.clf()
plt.scatter(unif_X_tra[unif_y_tra==1,0], unif_X_tra[unif_y_tra==1,1], c="Blue", label='y = 1')
plt.scatter(unif_X_tra[unif_y_tra==-1,0], unif_X_tra[unif_y_tra==-1,1], c="Red", label='y = -1')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Muestra aleatoria uniforme de 1000 elementos etiquetados con ruido")
plt.legend()
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# c.
print("Experimento: muestra aleatorias de 1000")
# De la misma forma que en el apartado anterior, añadimos
# un 1 a todos los vectores de características,
# tal y como se pide en el enunciado
unif_X_tra_ones = add_ones_column(unif_X_tra)


# Ajustamos un modelo de regresión lineal con el 
# Gradiente descendiente Estocástico

np.random.seed(12345)
w_ini = np.random.random(3)

# Máximas iteraciones para este experimento
max_it = 1000

w_unif, it = sgd(squared_error_gradient,mean_squared_error, w_ini, unif_X_tra_ones, unif_y_tra, max_iterations=max_it,random_seed=12345)

e_in_rand = mean_squared_error(w_unif,unif_X_tra_ones, unif_y_tra)

print("Ajuste obtenido con SGD para la muestra aleatoria de 1000 elementos:")
print(w_unif)
print("Error DENTRO de la muestra aleatoria: %f" % e_in_rand)


print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# d.
print("Experimento: 1000 muestras aleatorias de 1000")
np.random.seed(12345)

# Listas que guardará los errores dentro y
# fuera de la muestra correspondientemente
errors_in_list = []
errors_out_list = []

# Repetimos el experimento anterior 1000 veces
for j in range(0,1000,1):
    print("Iteracion %d del experimento..." % j)
    # Generamos training
    unif_X_tra = simula_unif(1000,2,1.0)
    unif_y_tra = f_label_vectorized(unif_X_tra[:,0],unif_X_tra[:,1])
    unif_X_tra_ones = add_ones_column(unif_X_tra)
    
    for i in range(0,len(unif_y_tra)//10,1):
        rand_i = random.randint(0,len(unif_y_tra)-1)
        unif_y_tra[rand_i] = -unif_y_tra[rand_i]
        
    w_unif, it = sgd(squared_error_gradient,mean_squared_error, w_ini, unif_X_tra_ones, unif_y_tra, max_iterations=max_it,random_seed=12345)
    e_in_rand = mean_squared_error(w_unif,unif_X_tra_ones, unif_y_tra)
    errors_in_list.append(e_in_rand)
    
    # Generamos test
    unif_X_tst = simula_unif(1000,2,1.0)
    unif_y_tst = f_label_vectorized(unif_X_tst[:,0],unif_X_tst[:,1])
    unif_X_tst_ones = add_ones_column(unif_X_tst)
    
    for i in range(0,len(unif_y_tst)//10,1):
        rand_i = random.randint(0,len(unif_y_tst)-1)
        unif_y_tst[rand_i] = -unif_y_tst[rand_i]
    
    e_out_rand = mean_squared_error(w_unif,unif_X_tst_ones, unif_y_tst)
    errors_out_list.append(e_out_rand)

errors_in = np.array(errors_in_list)
average_e_in = np.mean(errors_in)

errors_out = np.array(errors_out_list)
average_e_out = np.mean(errors_out)


print("Media de errores DENTRO de la muestra: %f" % average_e_in)
print("Media de errores FUERA de la muestra: %f" % average_e_out)




# BONUS

# Función para obtener la matriz Hessiana de la función f(x,y)
# previamente definida
def hessian_f(xy):
    x = xy[0]
    y = xy[1]
    
    # Segundas derivadas parciales
    dxx = 2 - 8*math.pi*math.pi*math.sin(2*math.pi*y)*math.sin(2*math.pi*x)
    dxy = 8*math.pi*math.pi*math.cos(2*math.pi*y)*math.cos(2*math.pi*x)
    dyy = 4 - 8*math.pi*math.pi*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

    hessian = np.array([[dxx,dxy],[dxy,dyy]])
    
    return hessian



# Implementación del método de Newton
# Similar al gradiente descendiente, pero usando la Hessiana
# además del coeficiente eta (learning rate)
# Quitamos también la precisión como criterio de parada
def newtons_method(gradient, hessian, starting_point, max_iterations=100000,
                   learning_rate=1.0):
    current_point = starting_point.copy()
    iterations = 0
    gradient_value = gradient(current_point)
    hessian_value = hessian(current_point)
    inv_hessian = np.linalg.inv(hessian_value)
    
    points = []
    points.append(current_point.copy())

    while iterations < max_iterations:
        # Actualizamos el punto actual
        current_point = current_point.copy() - learning_rate*np.dot(inv_hessian,gradient_value)
        iterations += 1
        # Actualizamos el gradiente y la inversa de la Hessiana
        gradient_value = gradient(current_point)
        hessian_value = hessian(current_point)
        inv_hessian = np.linalg.inv(hessian_value)
        
        points.append(current_point.copy())
        
    
    points_np = np.array(points)
    
    return current_point, iterations, points_np, 

# Punto de inicio: (0.1, 0.1)
starting_point_f = np.array([0.1,0.1])

op, it, pts = newtons_method(df, hessian_f, starting_point_f.copy(), 50)


print("Mínimo de f(x,y) encontrado con método de Newton tras 50 iteraciones: %f" % f(op[0],op[1]))
print("Coordenadas del punto:")
print(op)
print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()



# Imprimimos cómo evoluciona el valor de f(x,y)



# Vectorizamos la función f(x,y) para poder aplicarla directamente sobre 
# el grid de las componentes 'x' e 'y' para obtener las componentes 'z'
vectorized_f = np.vectorize(f)
x_points = pts[:,0]
y_points = pts[:,1]
f_points = vectorized_f(x_points, y_points)
iterations = np.arange(0,len(pts),1)

plt.clf()
plt.plot(iterations,f_points)
plt.xlabel("Iteración")
plt.ylabel("f(x,y)")
plt.title("Valores de f(x,y) explorados en Método de Newton")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

op, it, pts = newtons_method(df, hessian_f, starting_point_f.copy(), 50, 0.01)

print("Mínimo de f(x,y) encontrado con método de Newton tras 50 iteraciones (learning rate 0.01): %f" % f(op[0],op[1]))
print("Coordenadas del punto:")
print(op)
print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


x_points = pts[:,0]
y_points = pts[:,1]
f_points = vectorized_f(x_points, y_points)
iterations = np.arange(0,len(pts),1)

plt.clf()
plt.plot(iterations,f_points)
plt.xlabel("Iteración")
plt.ylabel("f(x,y)")
plt.title("Valores de f(x,y) explorados en Método de Newton (learning rate 0.01)")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

op, it, pts = newtons_method(df, hessian_f, starting_point_f.copy(), 50, 0.1)

print("Mínimo de f(x,y) encontrado con método de Newton tras 50 iteraciones (learning rate 0.1): %f" % f(op[0],op[1]))
print("Coordenadas del punto:")
print(op)
print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

x_points = pts[:,0]
y_points = pts[:,1]
f_points = vectorized_f(x_points, y_points)
iterations = np.arange(0,len(pts),1)

plt.clf()
plt.plot(iterations,f_points)
plt.xlabel("Iteración")
plt.ylabel("f(x,y)")
plt.title("Valores de f(x,y) explorados en Método de Newton (learning rate 0.1)")
plt.show()

print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()


# Lista con puntos de inicio
starting_points = [np.array([0.1,0.1]),np.array([1.0,1.0]),
                   np.array([-0.5,-0.5]),np.array([-1.0,-1.0])]

# Guardamos los valores mínimos y las coordenadas de sus respectivos puntos
minimums = []
points = []

# Para cada punto de inicio definido, ejecutamos el gradiente descendente
# para saber qué mínimo se alcanza y en qué punto
for s_pt in starting_points:
    p, it, pts = newtons_method(df, hessian_f, s_pt.copy(), 50, 0.01)
    points.append(p)
    minimums.append(f(p[0],p[1]))

# Una vez tenemos los resultados para distintos puntos de inicio,
# los imprimimos en una 'tabla' por pantalla

print("Mínimos de f(x,y) obtenidos a partir de distintos puntos de inicio con el método de Newton(learning rate de 0,01):\n")

print("Punto de inicio\tValor mínimo\tCoordenadas de (x,y)")
print("---------------------------------------------------")

for i in range(0,len(starting_points),1):
    p = points[i]
    m = minimums[i]
    stp = starting_points[i]
    print("%s   \t%f  \t%s" % (np.array2string(stp,precision=2, separator=','), m,
                               np.array2string(p,precision=2, separator=',')))


print("\n******************************")
print("Pulse Enter para continuar")
print("******************************\n")
input()

"""
    FIN DE LA PRÁCTICA 1
"""