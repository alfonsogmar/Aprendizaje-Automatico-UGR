# -*- coding: utf-8 -*-
"""
TRABAJO 3. Ajuste de Modelos Lineales
    Problema de clasificación (Optical Recognition of Handwritten Digits)
    Alfonso García Martínez 
"""

import numpy as np
import matplotlib.pylab as plt
import math
#import random as rnd
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

import seaborn as sns
import pandas as pd



"""
    CARGAR EL DATASET 'HANDWRITTEN DIGITS'
"""

# Lectura de datos desde los ficheros
digits_tra = np.loadtxt("datos/optdigits.tra",delimiter=",")
digits_tst = np.loadtxt("datos/optdigits.tes",delimiter=",")

# Separar etiquetas (y) de características (X)
digits_tra_X = digits_tra[:,0:len(digits_tra[0])-1]
digits_tra_y = digits_tra[:,-1]

digits_tst_X = digits_tst[:,0:len(digits_tst[0])-1]
digits_tst_y = digits_tst[:,-1]


"""
    EXPLORACIÓN DEL CONJUNTO DE DATOS
"""
# Solo estudiar conjunto de training


# Distribución de valores de las etiquetas
plt.hist(digits_tra_y, edgecolor='white', linewidth=1.2)
plt.title("Distribución de valores de etiquetas (digit)")
plt.xlabel("Dígito escrito")
plt.ylabel("Nº instancias")
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


print()
print("Matriz de correlación de digits")
print()

digits_corr = np.corrcoef(digits_tra_X.T)
digits_corr_df = pd.DataFrame(digits_corr)
ax = sns.heatmap(digits_corr_df, linewidth=0.5)
plt.xticks(rotation=30)
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()



# Calcula el número de instancias mal clasificadas a partir
# de la exactitud y el número total de instancias
def missclassified_samples(accuracy, n_samples):
    return (1.0-accuracy)*n_samples

# Calcular exactitud con validación cruzada estratificada
# Para problemas de clasificación balanceados
def stratifiedKFoldCrossValidationAccuracy(model, X, y, cv):
    y_test_all = []
    performance = 0

    for train, test in cv.split(X, y):
        #t = time.time()
        model = model.fit(X[train],y[train])
        #tiempo = time.time() - t
        y_pred = model.predict(X[test])

        perf = accuracy_score(y[test],y_pred)
        performance += perf
        #print("Error dentro de CV %.3f" % err)
        y_test_all = np.concatenate([y_test_all,y[test]])

    performance =performance/cv.get_n_splits()
    return y_test_all, performance

# Calcular f1 con validación cruzada estratificada
# Para problemas de clasificación balanceados
def stratifiedKFoldCrossValidationF1(model, X, y, cv):
    y_test_all = []
    performance = 0

    for train, test in cv.split(X, y):
        #t = time.time()
        model = model.fit(X[train],y[train])
        #tiempo = time.time() - t
        y_pred = model.predict(X[test])

        perf = accuracy_score(y[test],y_pred)
        performance += perf
        #print("Error dentro de CV %.3f" % err)
        y_test_all = np.concatenate([y_test_all,y[test]])

    performance =performance/cv.get_n_splits()
    return y_test_all, performance

print()
print("Perceptrón simple")
print()


perc = Perceptron(random_state=12345, max_iter=50000, tol=10**(-14))

perc = perc.fit(digits_tra_X,digits_tra_y)
digits_tra_y_pred = perc.predict(digits_tra_X)
accu = accuracy_score(digits_tra_y, digits_tra_y_pred)
f1 = f1_score(digits_tra_y, digits_tra_y_pred, average='weighted')
clasif_error = missclassified_samples(accu, len(digits_tra_y))


perc = Perceptron(random_state=12345, max_iter=50000, tol=10**(-14))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, accu_cv = stratifiedKFoldCrossValidationAccuracy(perc, digits_tra_X, digits_tra_y, skf)

perc = Perceptron(random_state=12345, max_iter=50000, tol=10**(-14))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, f1_cv = stratifiedKFoldCrossValidationF1(perc, digits_tra_X, digits_tra_y, skf)

clasif_error_cv = missclassified_samples(accu_cv, len(digits_tra_y))

print("Exactitud perceptrón en training: %.4f" % accu)
print("Ejemplos mal clasificados en training: %d" % clasif_error)
print("F1 en training: %.4f" % f1)
print()
print("Exactitud perceptrón en validación: %.4f" % accu_cv)
print("Ejemplos mal clasificados en validación: %d" % clasif_error_cv)
print("F1 en validación: %.4f" % f1_cv)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("Perceptrón simple con normalización")
print()

scaler = MinMaxScaler()
airfoil_X_tra_scaled = scaler.fit_transform(digits_tra_X)
airfoil_X_tra_non_scaled = digits_tra_X.copy()
digits_tra_X = airfoil_X_tra_scaled.copy()

perc = Perceptron(random_state=12345, max_iter=50000, tol=10**(-14))

perc = perc.fit(digits_tra_X,digits_tra_y)
digits_tra_y_pred = perc.predict(digits_tra_X)
accu = accuracy_score(digits_tra_y, digits_tra_y_pred)
f1 = f1_score(digits_tra_y, digits_tra_y_pred, average='weighted')
clasif_error = missclassified_samples(accu, len(digits_tra_y))


perc = Perceptron(random_state=12345, max_iter=50000, tol=10**(-14))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, accu_cv = stratifiedKFoldCrossValidationAccuracy(perc, digits_tra_X, digits_tra_y, skf)

perc = Perceptron(random_state=12345, max_iter=50000, tol=10**(-14))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, f1_cv = stratifiedKFoldCrossValidationF1(perc, digits_tra_X, digits_tra_y, skf)

clasif_error_cv = missclassified_samples(accu_cv, len(digits_tra_y))

print("Exactitud perceptrón en training: %.4f" % accu)
print("Ejemplos mal clasificados en training: %d" % clasif_error)
print("F1 en training: %.4f" % f1)
print()
print("Exactitud perceptrón en validación: %.4f" % accu_cv)
print("Ejemplos mal clasificados en validación: %d" % clasif_error_cv)
print("F1 en validación: %.4f" % f1_cv)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("Regresión logística (con normalización)")
print()



log_reg = LogisticRegression(solver='newton-cg',tol=10**(-8),multi_class='multinomial')

log_reg = log_reg.fit(digits_tra_X,digits_tra_y)
digits_tra_y_pred = log_reg.predict(digits_tra_X)
accu = accuracy_score(digits_tra_y, digits_tra_y_pred)
f1 = f1_score(digits_tra_y, digits_tra_y_pred, average='weighted')
clasif_error = missclassified_samples(accu, len(digits_tra_y))

# Validación
log_reg = LogisticRegression(solver='newton-cg',tol=10**(-8),multi_class='multinomial')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, accu_cv = stratifiedKFoldCrossValidationAccuracy(log_reg, digits_tra_X, digits_tra_y, skf)

log_reg = LogisticRegression(solver='newton-cg',tol=10**(-8),multi_class='multinomial')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, f1_cv = stratifiedKFoldCrossValidationF1(log_reg, digits_tra_X, digits_tra_y, skf)

clasif_error_cv = missclassified_samples(accu_cv, len(digits_tra_y))

accu_tra = accu

print("Exactitud perceptrón en training: %.4f" % accu)
print("Ejemplos mal clasificados en training: %d" % clasif_error)
print("F1 en training: %.4f" % f1)
print()
print("Exactitud perceptrón en validación: %.4f" % accu_cv)
print("Ejemplos mal clasificados en validación: %d" % clasif_error_cv)
print("F1 en validación: %.4f" % f1_cv)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("Regresión logística con selección de características por varianza")
print()

# Eliminamos características con varianza menor que 0.1
feat_sel = VarianceThreshold(0.05)
digits_tra_X_thres = feat_sel.fit_transform(digits_tra_X)

log_reg = LogisticRegression(solver='newton-cg',tol=10**(-8),multi_class='multinomial')

log_reg = log_reg.fit(digits_tra_X_thres,digits_tra_y)
digits_tra_y_pred = log_reg.predict(digits_tra_X_thres)
accu = accuracy_score(digits_tra_y, digits_tra_y_pred)
f1 = f1_score(digits_tra_y, digits_tra_y_pred, average='weighted')
clasif_error = missclassified_samples(accu, len(digits_tra_y))

# Validación
log_reg = LogisticRegression(solver='newton-cg',tol=10**(-8),multi_class='multinomial')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, accu_cv = stratifiedKFoldCrossValidationAccuracy(log_reg, digits_tra_X_thres, digits_tra_y, skf)

log_reg = LogisticRegression(solver='newton-cg',tol=10**(-8),multi_class='multinomial')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
y_cv, f1_cv = stratifiedKFoldCrossValidationF1(log_reg, digits_tra_X_thres, digits_tra_y, skf)

clasif_error_cv = missclassified_samples(accu_cv, len(digits_tra_y))

print("Exactitud perceptrón en training: %.4f" % accu)
print("Ejemplos mal clasificados en training: %d" % clasif_error)
print("F1 en training: %.4f" % f1)
print()
print("Exactitud perceptrón en validación: %.4f" % accu_cv)
print("Ejemplos mal clasificados en validación: %d" % clasif_error_cv)
print("F1 en validación: %.4f" % f1_cv)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("Rendimiento en test")
print()

scaler = MinMaxScaler()
digits_tst_X = scaler.fit_transform(digits_tst_X)

log_reg = LogisticRegression(solver='newton-cg',tol=10**(-8),multi_class='multinomial')

# Ajustamos con training
log_reg = log_reg.fit(digits_tra_X,digits_tra_y)

# Predecimos test
digits_tst_y_pred = log_reg.predict(digits_tst_X)
accu = accuracy_score(digits_tst_y, digits_tst_y_pred)
f1 = f1_score(digits_tst_y, digits_tst_y_pred, average='weighted')
clasif_error = missclassified_samples(accu, len(digits_tst_y))

print("Exactitud perceptrón en test: %.4f" % accu)
print("Ejemplos mal clasificados en test: %d" % clasif_error)
print("F1 en test: %.4f" % f1)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("ESTIMACÓN Eout")
print()

err = 1.0-accu
delta = 0.05
prob = 1.0-delta
e_out_test = err + math.sqrt((1/(2*len(digits_tst_y)))*math.log(2/delta))
print("Cota de Eout estimada a partir de test: %.4f" % e_out_test)
print("Con una probabilidad del %.4f" % prob)

err = 1.0-accu_tra
d_vc = len(digits_tra_X[0])
N=len(digits_tra_y)
e_out_tra = err + math.sqrt((8/N)*math.log((4*(((2*N)**d_vc)+1)/delta)))
