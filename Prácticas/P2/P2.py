# -*- coding: utf-8 -*-
"""
TRABAJO 2. 
    Alfonso García Martínez 
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rnd

# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)   
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

"""
    1.- Ejercicio sobre la complejidad de H y el ruido
"""

print("EJERCICIO 1\n")

# 1.- Dibujar una gráfica con la nube de puntos de salida correspondiente.

print("1.")

np.random.seed(12345)

unif = simula_unif(50,2,[-50,50])
gaus = simula_gaus(50,2,[5,7])

plt.clf

plt.scatter(unif[:,0],unif[:,1],color="Blue")
plt.title("Distrib. Uniforme: simula_unif(50,2,[-50,50])")
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

plt.scatter(gaus[:,0],gaus[:,1],color="Orange")
plt.title("Distrib. Gaussiana: simula_gaus(50,2,[5,7])")
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

plt.scatter(unif[:,0],unif[:,1],color="Blue",label="Distrib. Uniforme: simula_unif(50,2,[-50,50])")
plt.scatter(gaus[:,0],gaus[:,1],color="Orange",label="Distrib. Gaussiana: simula_gaus(50,2,[5,7])")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("2.")
print("a.")
def f_sign(x,y,a,b):
    return math.copysign(1,y - a*x - b)

X = simula_unif(200,2,[-50,50])
f_vectorized = np.vectorize(f_sign)
param_a, param_b = simula_recta([-50,50])

print("Parámetros de la recta aleatoria y = ax + b")
print("a = %.3f" % param_a)
print("b = %.3f" % param_b)


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


y = f_vectorized(X[:,0],X[:,1],param_a,param_b)

t = np.arange(-50,50,5)

plt.scatter(X[y==1,0],X[y==1,1],color="Blue",label="+1")
plt.scatter(X[y==-1,0],X[y==-1,1],color="Red",label="-1")
plt.plot(t, t*param_a + param_b,color="Green",label=("y = %.2fx+%.2f" % (param_a, param_b)))
plt.title("Distribución uniforme de puntos con recta separadora")
plt.legend()
plt.show()



print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("b.")

# Metemos ruido aleatorio en un 10% de la muestra
rnd.seed(12345)
noisy_y = y.copy()


unique, positives = np.unique(noisy_y, return_counts=True)
print(unique)
print(positives)

# Guardamos en esta lista los índices de los elementos 
# cuyos signos ya hemos cambiado
switched_labels = []

# Cambiamos 10% de las etiquetas positivas
i=0
while i<(positives[1]//10):
    rand_indx = rnd.randint(0,len(noisy_y)-1)
    if noisy_y[rand_indx] == 1 and rand_indx not in switched_labels:
        noisy_y[rand_indx]=-1
        i = i+1
        switched_labels.append(rand_indx)


# Cambiamos 10% de las etiquetas negativas
i=0
while i<(positives[0]//10):
    rand_indx = rnd.randint(0,len(noisy_y)-1)
    if noisy_y[rand_indx] == -1 and rand_indx not in switched_labels:
        noisy_y[rand_indx]=1
        i = i+1
        switched_labels.append(rand_indx)

del switched_labels

print("2.")

# Volvemos a pintar
plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t, t*param_a + param_b,color="Green",label=("y = %.2fx+%.2f" % (param_a, param_b)))
plt.title("Distribución uniforme de puntos con recta separadora y ruido en etiquetas")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("3.")

# Intersecciones de las funciones f(x,y) = 0
def f1_m(x):
    return (20 - math.sqrt(300 + 20*x - x**2))
def f1_p(x):
    return (20 + math.sqrt(300 + 20*x - x**2))

def f2_m(x):
    return 1/2*( 40 - math.sqrt(2)*math.sqrt(-x**2 - 20*x + 700))
def f2_p(x):
    return 1/2*( 40 + math.sqrt(2)*math.sqrt(-x**2 - 20*x + 700))

def f3_m(x):
    return 1/2 * ( -40 - math.sqrt(2) * math.sqrt(x**2 - 20*x - 700))
def f3_p(x):
    return 1/2 * ( -40 + math.sqrt(2) * math.sqrt(x**2 - 20*x - 700))

def f4(x):
    return 20*x**2 + 5*x - 3


# Primera función f(x,y)


f1_mv = np.vectorize(f1_m)
f1_pv = np.vectorize(f1_p)

# Rango de valores para los que 'f1_m' y 'f1_p' deben pintarse
# (estamos asegurándonos de que dentro de la raíz cuadrada
# nunca haya un valor negativo)
#t = np.arange(-10,31,1)
t = np.linspace(-10,30,50)

# Para igualar las escalas de los ejes (no lo usaremos)
#plt.xlim(-50, 50)
#plt.ylim(-50, 50)
#plt.gca().set_aspect('equal', adjustable='box')

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t, f1_mv(t),color="Green",label="y = 20 +- sqrt(300 + 20*x - x^2)")
plt.plot(t, f1_pv(t),color="Green")
plt.title("Distribución uniforme de puntos con curva separadora 1 y ruido en etiquetas")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Segunda función f(x,y)

f2_mv = np.vectorize(f2_m)
f2_pv = np.vectorize(f2_p)

# Rango en el cual el contenido de las raices cuadradas de f2_m y f2_p 
# será positivo
t = np.linspace(-38.28, 18.28,50)

# Pintamos los datos con la curva separadora
plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t, f2_mv(t),color="Green",label="y = 1/2 ( 40 +- sqrt(2)*sqrt(-x^2 - 20x + 700))")
plt.plot(t, f2_pv(t),color="Green")
plt.title("Distribución uniforme de puntos con curva separadora 2 y ruido en etiquetas")
plt.legend()
plt.show()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Tercera función f(x,y)

f3_mv = np.vectorize(f3_m)
f3_pv = np.vectorize(f3_p)

# Rangos en los cuales el contenido de las raices cuadradas de f3_m y f3_p 
# serán positivos
t1 = np.linspace(-50.0,-18.29,40)
t2 = np.linspace(38.29,50.0,40)

# Pintamos los datos con la tercera curva separadora
plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t1, f3_mv(t1),color="Green",label="y = 1/2 ( -40 +-sqrt(2)*sqrt(x^2 - 20x - 700))")
plt.plot(t1, f3_pv(t1),color="Green")
plt.plot(t2, f3_mv(t2),color="Green")
plt.plot(t2, f3_pv(t2),color="Green")
plt.title("Distribución uniforme de puntos con curva separadora 3 (hipérbola) y ruido en etiquetas")
plt.legend()
plt.show()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Cuarta función f(x,y)

f4v = np.vectorize(f4)

# No tenemos límite en el rango, fijamos el mismo
# en el que se han generado los puntos
t = np.linspace(-50.0,50.0,40);


plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t, f4v(t),color="Green",label="y = 20x^2 + 5x - 3")
plt.title("Distribución uniforme de puntos con curva separadora 4 (parábola) y ruido en etiquetas")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Volvemos a pintar los puntos restringiendo el dominio de la cuarta curva
t = np.linspace(-1.76 , 1.51,10)
plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t, f4v(t),color="Green",label="y = 20x^2 + 5x - 3")
plt.title("Distribución uniforme de puntos con curva separadora 4 (parábola) con dominio restringido")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()
#######################################################################

print("EJERCICIO 2\n")

print("1.-")
#1. Algoritmo Perceptrón
def ajusta_PLA(datos, label, max_iter, vini):
    lin_model = vini.copy()
    misclassified = True
    iterations = 0
    # Mientras encontremos instancias mal clasificadas
    while misclassified and iterations < max_iter:
        misclassified = False
        for i in range(0,len(label),1):
            data = datos[i]
            # Calculamos el signo
            sign = math.copysign(1,np.dot(lin_model,data))
            #sign = math.copysign(1,np.dot(data,lin_model))
            if(label[i] != sign):
                # Indicamos que hay una instancia mal clasificada,
                # luego habrá que volver a recorrer el conjunto de datos
                misclassified = True
                # Regla de actualización del perceptrón
                lin_model = lin_model + label[i]*data
        iterations = iterations + 1
    return lin_model, iterations


# Función para añadir una columna de 1's a los datos,
# para poder ajustar también el término independiente
def add_ones_column(data):
    ones = np.full((len(data),1),1.0)
    concatenated = np.concatenate((ones,data),axis=1)
    return concatenated


# a. PLA con conjunto de datos etiquetados sin ruido
print("a.")


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

t = np.linspace(-50.0,50.0,40);

X_ones = add_ones_column(X)
w_init = np.zeros(3)
w_1a,it = ajusta_PLA(X_ones, y, 1000,w_init)

plt.scatter(X[y==1,0],X[y==1,1],color="Blue",label="+1")
plt.scatter(X[y==-1,0],X[y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1a[0]/w_1a[2]-w_1a[1]/w_1a[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1a[0]/w_1a[2],-w_1a[1]/w_1a[2])))
plt.title("Puntos sin ruido separados por perceptrón ajustado con situ. inicial w=0")
plt.legend()
plt.show()
print("Separación del conjunto de datos para valor inicial w=0 lograda en %d iteraciones" % it)


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Ajustamos 10 veces con w iniciales con valores aleatorios entre [0,1]

iterations_list = [it]
initial_values = [w_init.copy()]
fitted = [w_1a.copy()]

for k in range(0,10,1):
    w_init = np.random.rand(3)
    w_1a, it = ajusta_PLA(X_ones, y, 1000,w_init)
    iterations_list.append(it)
    initial_values.append(w_init.copy())
    fitted.append(w_1a.copy())
    
    
# Con los datos que acabamos de guardar, imprimimos por consola una tabla
print("Ajustes obtenidos con PLA a partir de distintos puntos de inicio:\n")

print("Punto de inicio\t\tAjuste obtenido\t\t\tIteraciones")
print("----------------------------------------------------------------------")

for i in range(0,len(initial_values),1):
    init = initial_values[i]
    w = fitted[i]
    itrs = iterations_list[i]
    print("%s      \t%s  \t%i" % (np.array2string(init,precision=2, separator=','),
          np.array2string(w,precision=2, separator=','),itrs))
print("----------------------------------------------------------------------")

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

itrs_array = np.array(iterations_list)

print("Número medio de iteraciones para los 11 valores iniciales iniciales: %d" %
      itrs_array.mean())

iterations_list.pop(0)
itrs_array = np.array(iterations_list)

print("Número medio de iteraciones para los 10 iniciales aleatorios (sin contar inicial w=0): %d" %
      itrs_array.mean())

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

w_1a_rand3 = fitted[3]
w_1a_rand5 = fitted[5]
w_1a_rand7 = fitted[7]

plt.scatter(X[y==1,0],X[y==1,1],color="Blue",label="+1")
plt.scatter(X[y==-1,0],X[y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1a_rand3[0]/w_1a_rand3[2]-w_1a_rand3[1]/w_1a_rand3[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1a_rand3[0]/w_1a_rand3[2],-w_1a_rand3[1]/w_1a_rand3[2])))
plt.title("Puntos sin ruido separados por perceptrón ajustado con 3ª situ. inicial aleatoria")
plt.legend()
plt.show()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


plt.scatter(X[y==1,0],X[y==1,1],color="Blue",label="+1")
plt.scatter(X[y==-1,0],X[y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1a_rand5[0]/w_1a_rand5[2]-w_1a_rand5[1]/w_1a_rand5[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1a_rand5[0]/w_1a_rand5[2],-w_1a_rand5[1]/w_1a_rand5[2])))
plt.title("Puntos sin ruido separados por perceptrón ajustado con 5ª situ. inicial aleatoria")
plt.legend()
plt.show()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


plt.scatter(X[y==1,0],X[y==1,1],color="Blue",label="+1")
plt.scatter(X[y==-1,0],X[y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1a_rand7[0]/w_1a_rand7[2]-w_1a_rand7[1]/w_1a_rand7[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1a_rand7[0]/w_1a_rand7[2],-w_1a_rand7[1]/w_1a_rand7[2])))
plt.title("Puntos sin ruido separados por perceptrón ajustado con 7ª situ. inicial aleatoria")
plt.legend()
plt.show()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


# b. PLA con conjunto de datos CON ruido
print("b.")



# Intentamos ajustar el conjunto de datos ruidosos con el PLA
# para un valor inicial w=0
X_ones = add_ones_column(X)
w_init = np.zeros(3)
w_1b,it = ajusta_PLA(X_ones, noisy_y, 1000,w_init)

# Pintamos
t = np.linspace(-50.0,50.0,40);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b[0]/w_1b[2]-w_1b[1]/w_1b[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b[0]/w_1b[2],-w_1b[1]/w_1b[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con situ. inicial w=0")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

t = np.linspace(-27.0,-2.0,20);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b[0]/w_1b[2]-w_1b[1]/w_1b[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b[0]/w_1b[2],-w_1b[1]/w_1b[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con situ. inicial w=0")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("Separación del conjunto de datos para valor inicial w=0 lograda en %d iteraciones" % it)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

iterations_list_noisy = []
fitted_noisy = []

for initial in initial_values:
    w_1b, it = ajusta_PLA(X_ones, noisy_y, 1000, initial)
    iterations_list_noisy.append(it)
    fitted_noisy.append(w_1b.copy())
    
    
# Con los datos que acabamos de guardar, imprimimos por consola una tabla
print("Ajustes obtenidos con PLA a partir de distintos puntos de inicio:\n")

print("Punto de inicio\t\tAjuste obtenido\t\t\tIteraciones")
print("----------------------------------------------------------------------")

for i in range(0,len(initial_values),1):
    init = initial_values[i]
    w = fitted_noisy[i]
    itrs = iterations_list_noisy[i]
    print("%s      \t%s  \t%i" % (np.array2string(init,precision=2, separator=','),
          np.array2string(w,precision=2, separator=','),itrs))
print("----------------------------------------------------------------------")


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

w_1b_rand3 = fitted_noisy[3]
w_1b_rand5 = fitted_noisy[5]
w_1b_rand7 = fitted_noisy[7]

t = np.linspace(-50.0,50.0,40);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b_rand3[0]/w_1b_rand3[2]-w_1b_rand3[1]/w_1b_rand3[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b_rand3[0]/w_1b_rand3[2],-w_1b_rand3[1]/w_1b_rand3[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con 3ª situ. inicial aleatoria")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

t = np.linspace(-47.0,23.0,20);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b_rand3[0]/w_1b_rand3[2]-w_1b_rand3[1]/w_1b_rand3[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b_rand3[0]/w_1b_rand3[2],-w_1b_rand3[1]/w_1b_rand3[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con 3ª situ. inicial aleatoria (recta en intervalo restringido)")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

t = np.linspace(-50.0,50.0,40);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b_rand5[0]/w_1b_rand5[2]-w_1b_rand5[1]/w_1b_rand5[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b_rand5[0]/w_1b_rand5[2],-w_1b_rand5[1]/w_1b_rand5[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con 5ª situ. inicial aleatoria")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


t = np.linspace(-50.0,25.0,40);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b_rand5[0]/w_1b_rand5[2]-w_1b_rand5[1]/w_1b_rand5[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b_rand5[0]/w_1b_rand5[2],-w_1b_rand5[1]/w_1b_rand5[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con 5ª situ. inicial aleatoria (recta en intervalo restringido)")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

t = np.linspace(-50.0,50.0,40);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b_rand7[0]/w_1b_rand7[2]-w_1b_rand7[1]/w_1b_rand7[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b_rand7[0]/w_1b_rand7[2],-w_1b_rand7[1]/w_1b_rand7[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con 7ª situ. inicial aleatoria")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

t = np.linspace(-22.0,5.0,40);

plt.scatter(X[noisy_y==1,0],X[noisy_y==1,1],color="Blue",label="+1")
plt.scatter(X[noisy_y==-1,0],X[noisy_y==-1,1],color="Red",label="-1")
plt.plot(t,-w_1b_rand7[0]/w_1b_rand7[2]-w_1b_rand7[1]/w_1b_rand7[2]*t ,color="Green",label=("Recta separadora x2 = %.2f + %.2fx1" % (-w_1b_rand7[0]/w_1b_rand7[2],-w_1b_rand7[1]/w_1b_rand7[2])))
plt.title("Puntos con ruido separados por perceptrón ajustado con 7ª situ. inicial aleatoria (recta en intervalo restringido)")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


# 2. Regresión logística con SGD
print("2. Regresión Logística")

#a.


# Función para generar minibatches aleatorios de un determinado tamaño
# a partir del conjunto de datos
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
    
    for i in range(0,n_minibatches,1): 
        mini = data_x_y[minibatch_begin:minibatch_end]
        mini_x = mini[:,0:len(data_x[0])]
        mini_y = mini[:,-1]
        minibatches.append((mini_x,mini_y))
        minibatch_begin += minibatch_size
        minibatch_end  += minibatch_size
    
    return minibatches 


    
# Implementación del Gradiente Descendiente Estocástico
# siguiendo las condiciones indicadas en el guión en el guión
# (suponemos que la semilla aleatoria ya esta fijada)
def sgd(gradient, data_X, data_y, learning_rate=0.01, 
        max_epochs=50, precision=0.01, minibatch_size=32):
    current_point = np.zeros(len(data_X[0]))
    epoch = 0
    

    step_size = precision + 9999999.9999
    
    while epoch < max_epochs and step_size > precision:
        # Generar minibatches al comenzar cada época
        minibatches = generate_minibatches(data_X, data_y, minibatch_size)
        i = 0
        # Guardamos el punto actual para comprobar la precisión
        # al final de la época
        previous_point = current_point.copy()
        # Iteramos sobre los minibatches
        while i < len(minibatches):
            #print("i = %d" % i)
            minibatch_X = minibatches[i][0]
            #print("X")
            #print(minibatch_X)
            minibatch_y = minibatches[i][1]
            #print("y")
            #print(minibatch_y)
            gradient_value = gradient(minibatch_X,minibatch_y,current_point)
            current_point -= learning_rate*gradient_value
            # Siguiente minibatch
            i += 1

        step_size = np.linalg.norm(current_point-previous_point)
        epoch += 1
        
    return current_point, epoch


# Métrica del error usada en reg. logística 
# (opuesto del neperiano de la verosimilitud entre el tamaño del conjunto) 
def logitError(data, labels, lin_model):
    error = 0
    n = len(labels)
    for k in range(0,n,1):
        error += math.log(1 + math.exp(-labels[k]*np.dot(np.transpose(lin_model),data[k])))
    error = error/n
    return error

# Gradiente del error (opuesto del neperiano de la verosimilitud) 
# usado en regresión logística
def errorGradient(data, labels, lin_model):
    gradient = np.zeros(len(data[0]))
    n = len(labels)
    for k in range(0,n,1):
        #print("k = %d" % k)
        #print("data[k]")
        #print(data[k])
        #print("data[k]: %d" % len(data[k]))
        #print("lin_model: %d" % len(lin_model))
        gradient = gradient + (labels[k]*data[k])/(1+math.exp(labels[k]*np.dot(np.transpose(lin_model),data[k])))
    gradient = -gradient/n
    return gradient

#b


# Generamos el dataset en [0, 2]x[0, 2] que se pide en el enunciado
N=100
X2 = simula_unif(N, 2, [0,2])

t = np.linspace(0,2,10)

plt.scatter(X2[:,0],X2[:,1],color="Black")
plt.title("Muestra de 100 elementos con distribución uniforme")
plt.show()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Seleccionamos dos puntos al azar

selected_points = []

rand_point =  rnd.randint(0,N-1)
selected_points.append(rand_point)

rand_point =  rnd.randint(0,N-1)
# Nos aseguramos de que el segundo punto aleatorio no sea
# el mismo que el primero
while rand_point in selected_points:
    rand_point = rnd.randint(0,N-1)
    
selected_points.append(rand_point)

# Generamos la recta que pasa por esos dos puntos

# Pendiente de la recta

p0 = X2[selected_points[0]]
p1 = X2[selected_points[1]]

m = (p1[1] - p0[1])/(p1[0] - p0[0])
    
# Etiquetamos los datos en función de la recta generada
# Ecuación punto pendiente:
# y - y0 = m(x - x0)
# y = mx - mx0 + y0
# el término independiente de la ecuación de la recta es - mx0 + y0

a = m
b = -m*p0[0] + p0[1]


plt.scatter(X2[:,0],X2[:,1],color="Black",label="Muestra de 100 elementos con distribución uniforme")
plt.scatter(p0[0],p0[1],color="Orange",label="Puntos aleatorios por los que pasa la recta")
plt.scatter(p1[0],p1[1],color="Orange")
plt.title("Puntos uniformemente distribuidos, con dos puntos aleatorios seleccionados")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

plt.scatter(X2[:,0],X2[:,1],color="Black",label="Muestra de 100 elementos con distribución uniforme")
plt.scatter(p0[0],p0[1],color="Orange",label="Puntos aleatorios por los que pasa la recta")
plt.scatter(p1[0],p1[1],color="Orange")
plt.plot(t,t*a + b,color="Green", label=("Resta separadora %f.2x + %f.2 que pasa por puntos aleatorios" % (a,b)))
plt.title("Puntos uniformemente distribuidos y recta separadora")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


# Etiquetamos los puntos según esa recta

y2 = f_vectorized(X2[:,0],X2[:,1],a,b)

plt.scatter(X2[y2==1,0],X2[y2==1,1],color="Blue",label="y=+1")
plt.scatter(X2[y2==-1,0],X2[y2==-1,1],color="Red",label="y=-1")
plt.plot(t,t*a + b,color="Green", label=("Recta separadora %f.2x + %f.2" % (a,b)))
plt.title("Puntos uniformemente distribuidos etiquetados según recta recta separadora")
plt.legend()
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

X2_ones = add_ones_column(X2)

w_reg,it = sgd(errorGradient,X2_ones, y2, precision=0.01, minibatch_size=1, learning_rate=0.01)


error_logit=logitError(X2_ones, y2, w_reg)
print("Error de regresión logística en la muestra: %.3f" % error_logit)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


# Ahora, crearemos una muestra grande (N=2000) para aplicarle
# regresión logística y calcularle Eout

# Función de predicción de probabilidad de un dato
# a partir de los pesos ajustados por regresión logística
def predict_logit(data, lin_model):
    s = np.dot(np.transpose(lin_model),data)
    return math.exp(s)/(1+math.exp(s))

# Devuelve una etiqueta en función de la probabilidad de la clase
# La frontera de decisión está en el punto medio entre 0 y 1, o sea, 0.5
def map_probabilities(prob):
    if(prob>0.5):
        label = 1
    else:
        label = -1
    return label

N=2000
X2_tst = simula_unif(N, 2, [0,2])
X2_tst_ones = add_ones_column(X2_tst)

# Probabilidades generadas con regresión logística
probs = np.zeros(N)

for i in range(0,len(X2_tst_ones)):
    probs[i] = predict_logit(X2_tst_ones[i],w_reg)
    
map_probs_v = np.vectorize(map_probabilities)

# Asignamos etiquetas según probabilidades
y2_tst = map_probs_v(probs)

error_logit_tst = logitError(X2_tst_ones, y2_tst, w_reg)

print("Error de regresión logística en nuevos datos (fuera de muestra): %.3f" % error_logit_tst)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

plt.scatter(X2_tst[y2_tst==1,0],X2_tst[y2_tst==1,1],color="Blue",label="y=+1")
plt.scatter(X2_tst[y2_tst==-1,0],X2_tst[y2_tst==-1,1],color="Red",label="y=-1")
plt.title("Nuevo conjunto de 2000 puntos clasificados según probabilidades obtenidas con Regresión Logística")
plt.legend()
plt.show()



###############################################################################
###############################################################################