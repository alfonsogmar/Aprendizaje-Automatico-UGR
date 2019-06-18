# -*- coding: utf-8 -*-
"""
TRABAJO 3. Ajuste de Modelos Lineales
    Problema de regresión (airfoil self-noise)
    Alfonso García Martínez 
"""
import numpy as np
import matplotlib.pylab as plt
import math
#import random as rnd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


# Para pairplot
import seaborn as sns
import pandas as pd

"""
    CARGAR EL DATASET 'AIRFOIL'
"""

airfoil = np.loadtxt("datos/airfoil_self_noise.dat",delimiter="\t")

# Separar etiquetas (y) de características (X)
airfoil_X = airfoil[:,0:len(airfoil[0])-1]
airfoil_y = airfoil[:,-1]


"""
    EXPLORACIÓN DEL CONJUNTO DE DATOS
"""

# Detectar valores perdidos

#Obtenemos máscara de booleanos para valores perdidos
missing_data_mask = np.isnan(airfoil_X)
# Contamos si hay algún True
missing_data_unique, missing_data_counts = np.unique(missing_data_mask, return_counts=True)
print("\nEXPLORACIÓN DEL CONJUNTO DE DATOS\n")
print("Conteo de valores que no son NaN: %d" % missing_data_counts)
print("No hay valores NaN en el dataset")

del missing_data_mask, missing_data_counts, missing_data_unique

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Distribución de valores de las etiquetas
plt.hist(airfoil_y)
plt.title("Distribución de valores de etiquetas (airfoil)")
plt.xlabel("Presión acústica")
plt.ylabel("Nº instancias")
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

unique_airfoil_y, count_airfoil_y = np.unique(airfoil_y, return_counts=True)
airfoil_y_distrib = dict(zip(unique_airfoil_y, count_airfoil_y))

y_max_count = count_airfoil_y.max()

y_max_count_positions, = np.where(count_airfoil_y == count_airfoil_y.max())

print("Media de y: %.3f" % airfoil_y.mean())

print("\nValores más repetidos (%d apariciones): " % count_airfoil_y.max())

for i in y_max_count_positions:
    print(unique_airfoil_y[i])
    
del unique_airfoil_y, count_airfoil_y, airfoil_y_distrib, y_max_count_positions

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("Distribución de valores de X (airfoil)")
airfoil_columns = ['Frequency','Angle of attack','Chord length','Free-stream velocity','Suction side displacement thickness']
airfoil_rows = range(0,len(airfoil_y))
airfoil_dataframe = pd.DataFrame(airfoil_X, index=airfoil_rows, columns=airfoil_columns)
sns.set(style="ticks")
sns.pairplot(airfoil_dataframe)
plt.show()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("Matriz de correlación de airfoil")
print()

airfoil_corr = np.corrcoef(airfoil_X.T)
airfoil_corr_df = pd.DataFrame(airfoil_corr, index=airfoil_columns, columns=airfoil_columns)
ax = sns.heatmap(airfoil_corr_df, linewidth=0.5)
plt.xticks(rotation=30)
plt.show()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

"""
    PRIMER INTENTO
"""

# Probamos la regresión lineal tal cual
print("\nPRIMER INTENTO\n")
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X,airfoil_y)
airfoil_y_predict = lin_reg.predict(airfoil_X)

mse = mean_squared_error(airfoil_y, airfoil_y_predict)
mae = mean_absolute_error(airfoil_y, airfoil_y_predict)

print("Errores obtenidos (todos los datos)")
print("MSE: %.3f" % mse)
print("MAE: %.3f" % mae)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

"""
    VALIDACIÓN Y TEST
"""


print("\nVALIDACIÓN Y TEST\n")


# Validación cruzada
# Para problemas de regresión
def kFoldcrossValidation(model, X, y, cv, error_meas):
    y_test_all = []
    errors = [0] * len(error_meas)

    for train, test in cv.split(X, y):
        #t = time.time()
        model = model.fit(X[train],y[train])
        #tiempo = time.time() - t
        y_pred = model.predict(X[test])
        for i in range(0,len(error_meas)):
            err = error_meas[i](y[test],y_pred)
            errors[i] += err
            #print("Error dentro de CV %.3f" % err)
        y_test_all = np.concatenate([y_test_all,y[test]])
        
    for i in range(0,len(errors)):
        errors[i] = errors[i]/cv.get_n_splits()
    return y_test_all, errors


# Dividimos el dataset en training y test
airfoil_X_tra, airfoil_X_tst, airfoil_y_tra, airfoil_y_tst = train_test_split(
        airfoil_X, airfoil_y, test_size=0.2, random_state=12345)

# Usamos regresión lineal tal cual para este primer intento
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)

# Calculamos errores cuadrático y absoluto de test (no se hará más veces
# hasta que tengamos un modelo final)
airfoil_y_predict = lin_reg.predict(airfoil_X_tst)
mse_tst = mean_squared_error(airfoil_y_tst, airfoil_y_predict)
mae_tst = mean_absolute_error(airfoil_y_tst, airfoil_y_predict)

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en conjunto test)")
print("MSE: %.3f" % mse_tst)
print("MAE: %.3f" % mae_tst)
print()

# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


"""
    NORMALIZACIÓN
"""


print("\nNORMALIZACIÓN MINMAX\n")


scaler = MinMaxScaler()

airfoil_X_tra_scaled = scaler.fit_transform(airfoil_X_tra)


lin_reg_scaled = LinearRegression()
lin_reg_scaled = lin_reg.fit(airfoil_X_tra_scaled,airfoil_y_tra)

# DEBUG
#print("Con minmax")
#print(lin_reg_scaled.coef_)

airfoil_y_predict_minmax = lin_reg.predict(airfoil_X_tra_scaled)

mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict_minmax)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict_minmax)


lin_reg_scaled = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg_scaled,airfoil_X_tra_scaled,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

airfoil_X_tra_non_scaled = airfoil_X_tra.copy()
airfoil_X_tra = airfoil_X_tra_scaled.copy()
#del airfoil_X_tra_scaled


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()



"""
    SELECCIÓN DE CARACTERÍSTICAS
"""


print("\nSELECCIÓN DE CARACTERÍSTICAS: Umbral de varianza\n")

# En primer lugar, obtenemos la varianza de cada una de las características
feat_sel = VarianceThreshold()
feat_sel = feat_sel.fit(airfoil_X_tra)
variances = feat_sel.variances_

print("Varianzas de las características")
print("--------------------------------")
for i in range(0,len(variances)):
    print("%s: %.4f" % (airfoil_columns[i],variances[i]))
    
min_variance = variances.min()
print()
print("Eliminamos característica con varianza mínima de %.4f" % min_variance)
print()

# Eliminamos la variable con menos varianza
feat_sel = VarianceThreshold(min_variance+0.0005)
airfoil_X_tra_var_thres = feat_sel.fit_transform(airfoil_X_tra)


lin_reg_var_thres = LinearRegression()
lin_reg_var_thres = lin_reg.fit(airfoil_X_tra_var_thres,airfoil_y_tra)

# DEBUG
#print("Con minmax")
#print(lin_reg_scaled.coef_)

airfoil_y_predict_var_thres = lin_reg.predict(airfoil_X_tra_var_thres)

mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict_var_thres)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict_var_thres)


lin_reg_scaled = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg_scaled,airfoil_X_tra_var_thres,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()


print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()



print("\nSELECCIÓN DE CARACTERÍSTICAS: Eliminación manual de característica\n")
print()

print("Eliminamos manualmente la variable 'Angle of attack'")
print()

airfoil_X_tra_wo_angle = np.delete(airfoil_X_tra,1,axis=1)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_wo_angle,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_wo_angle)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_wo_angle,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()


print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()


print("Eliminamos manualmente la variable 'Suction side displacement thickness'")
print()

airfoil_X_tra_wo_thickness = np.delete(airfoil_X_tra,4,axis=1)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_wo_thickness,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_wo_thickness)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_wo_thickness,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("PCA 4 componentes")
print()
pca_reduc = PCA(4)
airfoil_X_tra_PCA = pca_reduc.fit_transform(airfoil_X_tra)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_PCA,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_PCA)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_PCA,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("PCA 3 componentes")
print()
pca_reduc = PCA(3)
airfoil_X_tra_PCA = pca_reduc.fit_transform(airfoil_X_tra)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_PCA,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_PCA)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_PCA,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("\nTRANSFORMACIONES NO LINEALES\n")
print()
print("\nPolinómica de órden 2\n")
print()
poly = PolynomialFeatures(2)

airfoil_X_tra_poly_2 = poly.fit_transform(airfoil_X_tra)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_poly_2,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_poly_2)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_poly_2,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("\nPolinómica de órden 3\n")
print()
poly = PolynomialFeatures(3)

airfoil_X_tra_poly_3 = poly.fit_transform(airfoil_X_tra)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_poly_3,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_poly_3)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_poly_3,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("\nPolinómica de órden 4\n")
print()
poly = PolynomialFeatures(4)

airfoil_X_tra_poly_4 = poly.fit_transform(airfoil_X_tra)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_poly_4,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_poly_4)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_poly_4,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("\nPolinómica de órden 5\n")
print()
poly = PolynomialFeatures(5)

airfoil_X_tra_poly_5 = poly.fit_transform(airfoil_X_tra)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_poly_5,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_poly_5)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_poly_5,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print("Intentando reducir complejidad de transf. polinómica (5) con PLA")

poly = PolynomialFeatures(5)
pca_reduc = PCA(100)

airfoil_X_tra_poly_5_PCA = poly.fit_transform(airfoil_X_tra)
airfoil_X_tra_poly_5_PCA = pca_reduc.fit_transform(airfoil_X_tra_poly_5)

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(airfoil_X_tra_poly_5_PCA,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_poly_5_PCA)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_poly_5_PCA,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("REGULARIZACIÓN")
print()

def ridge_changing_alpha_error(X,y):
    
    alpha_value = 1.0
    alphas_list = []
    mse_list = []
    while alpha_value > 10**(-6):
        model = Ridge(alpha=alpha_value)
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)
        model, ridge_errors = kFoldcrossValidation(model, X, y, kf,
                                                [mean_squared_error, mean_absolute_error])
        alphas_list.append(alpha_value)
        mse_list.append(ridge_errors[0])
        #print("alpha: %.8f" % alpha_value)
        #print("error: %.8f" % ridge_error)
        alpha_value = alpha_value*0.1
    alphas = np.array(alphas_list)
    errors = np.array(mse_list)
    plt.plot(alphas, errors, label="Error con CV")
    # Invertimos eje x para que la gráfica muestre de mayor a menor alfa
    plt.xlim(alphas.max(), alphas.min())
    plt.xlabel("Alfa")
    plt.ylabel("Error cuadrático medio")
    #print("alpha max: %.8f" % alphas.max())
    #print("alpha min: %.8f" % alphas.min())
    # Escala logarítmica para eje x (valores de alfa)
    plt.xscale('log')
    plt.title("Niveles obtenidos (con CV) para distintas alfas de regularización Ridge")
    plt.show()
    
def lasso_changing_alpha_error(X,y):
    
    alpha_value = 1.0
    alphas_list = []
    mse_list = []
    while alpha_value > 10**(-5):
        model = Lasso(alpha=alpha_value)
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)
        model, lasso_errors = kFoldcrossValidation(model, X, y, kf,
                                                [mean_squared_error, mean_absolute_error])
        alphas_list.append(alpha_value)
        mse_list.append(lasso_errors[0])
        #print("alpha: %.8f" % alpha_value)
        #print("error: %.8f" % ridge_error)
        alpha_value = alpha_value*0.1
    alphas = np.array(alphas_list)
    errors = np.array(mse_list)
    plt.plot(alphas, errors, label="Error con CV")
    # Invertimos eje x para que la gráfica muestre de mayor a menor alfa
    plt.xlim(alphas.max(), alphas.min())
    plt.xlabel("Alfa")
    plt.ylabel("Error cuadrático medio")
    #print("alpha max: %.8f" % alphas.max())
    #print("alpha min: %.8f" % alphas.min())
    # Escala logarítmica para eje x (valores de alfa)
    plt.xscale('log')
    plt.title("Niveles obtenidos (con CV) para distintas alfas de regularización Lasso")
    plt.show()
    
print("Regularización Ridge para airfoil original (normalizado)")
ridge_changing_alpha_error(airfoil_X_tra,airfoil_y_tra)
print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()
print("Regularización Ridge para airfoil transf. poli. orden 4")
ridge_changing_alpha_error(airfoil_X_tra_poly_4,airfoil_y_tra)
print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()
print("Regularización Ridge para airfoil transf. poli. orden 5")
ridge_changing_alpha_error(airfoil_X_tra_poly_5,airfoil_y_tra)
print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()
print("Regularización Lasso para airfoil transf. poli. orden 4")
lasso_changing_alpha_error(airfoil_X_tra_poly_4,airfoil_y_tra)
print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("Regularización Lasso para airfoil transf. poli. orden 5")
print()
lasso_changing_alpha_error(airfoil_X_tra_poly_5,airfoil_y_tra)

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

# Probamos reg. L5 junto con transformación poli. de orden 5

print()
print("Mejor regularización: L2 para trans. polinómica de orden 5")
print()

poly = PolynomialFeatures(5)

airfoil_X_tra_poly_5 = poly.fit_transform(airfoil_X_tra)

lin_reg = Ridge(alpha=10**(-5))
lin_reg = lin_reg.fit(airfoil_X_tra_poly_5,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tra_poly_5)

# Calculamos errores cuadrático y absoluto de training
mse_tra = mean_squared_error(airfoil_y_tra, airfoil_y_predict)
mae_tra = mean_absolute_error(airfoil_y_tra, airfoil_y_predict)


# Validación cruzada
lin_reg = Ridge(alpha=10**(-5))
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
airfoil_y_predict_cv, errors_cv = kFoldcrossValidation(lin_reg,airfoil_X_tra_poly_5,
                                                    airfoil_y_tra, kf,
                                                    [mean_squared_error,mean_absolute_error])

# Obtenemos errores cuadrático y absoluto de validación
mse_cv = errors_cv[0]
mae_cv = errors_cv[1]

print("Errores obtenidos (en conjunto training)")
print("MSE: %.3f" % mse_tra)
print("MAE: %.3f" % mae_tra)
print()

print("Errores obtenidos (en val. cruzada)")
print("MSE: %.3f" % mse_cv)
print("MAE: %.3f" % mae_cv)
print()

print("\n**************************")
print("Pulse Enter para continuar")
print("**************************")
input()

print()
print("CALCULAR ERROR DEL MEJOR MODELO EN TEST")
print()

# Aplicamos normalización y transformación polinómica a conjunto de test
scaler = MinMaxScaler()

airfoil_X_tst = scaler.fit_transform(airfoil_X_tst)

poly = PolynomialFeatures(5)

airfoil_X_tra_poly_5 = poly.fit_transform(airfoil_X_tra)
airfoil_X_tst_poly_5 = poly.fit_transform(airfoil_X_tst)

lin_reg = Ridge(alpha=10**(-5))
lin_reg = lin_reg.fit(airfoil_X_tra_poly_5,airfoil_y_tra)
airfoil_y_predict = lin_reg.predict(airfoil_X_tst_poly_5)

# Calculamos errores cuadrático y absoluto de test
mse_tst = mean_squared_error(airfoil_y_tst, airfoil_y_predict)
mae_tst = mean_absolute_error(airfoil_y_tst, airfoil_y_predict)

print("Errores test del mejor modelo:")
print("MSE: %.3f" % mse_tst)
print("MAE: %.3f" % mae_tst)

print()
print("ESTIMACÓN Eout")
print()

delta = 0.05
prob = 1.0-delta
e_out_test = mse_tst + math.sqrt((1/(2*len(airfoil_y_tst)))*math.log(2/delta))
print("Cota de Eout estimada a partir de test: %.4f" % e_out_test)
print("Con una probabilidad del %.4f" % prob)
