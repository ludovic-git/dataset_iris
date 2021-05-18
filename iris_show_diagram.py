import pandas
# import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
import sklearn

#traitement CSV
iris=pandas.read_csv("iris2.csv")
x=(iris.loc[:,"PetalLengthCm"])
x_scale = sklearn.preprocessing.scale(x)
# print(x_scale)

y=iris.loc[:,"PetalWidthCm"]
y_scale = sklearn.preprocessing.scale(y)

SL=iris.loc[:,"SepalLengthCm"]
SL_scale = sklearn.preprocessing.scale(SL)

SW=iris.loc[:,"SepalWidthCm"]
SW_scale = sklearn.preprocessing.scale(SW)

I = iris.loc[:"id"] 

lab=iris.loc[:,"Species"]


#fin traitement CSV

#valeurs
entree = float(input("longueur pétal ?"))    
longueur = entree
l_moy = longueur - x.mean() 
long_scale = l_moy/x.std()

entree = float(input("largeur pétal ?")) 
largeur = entree
la_moy = largeur - y.mean()
lar_scale = la_moy/y.std()

entree = float(input("longueur sépal ?")) 
l2=entree
ls = l2 - SL.mean()
slong = ls / SL.std()

entree = float(input("largeur sépal ?")) 
w2=entree
las= w2 - SW.mean()
slar = las = SW.std()

k=5

x_var= x_scale+ SL_scale
y_var =y_scale+SW_scale
#fin valeurs

#graphique
plt.subplot(3,1,1)
plt.axis('equal')
plt.scatter(x_var[lab == "Iris-setosa"], y_var[lab == "Iris-setosa"], color='g', label='setosa')
plt.scatter(x_var[lab == "Iris-versicolor"], y_var[lab == "Iris-versicolor"], color='r', label='versicolor')
plt.scatter(x_var[lab == "Iris-virginica"], y_var[lab == "Iris-virginica"], color='b', label='virginica')
plt.scatter(long_scale+slong, lar_scale+slar, color='k')
plt.xlabel("longueur pétal")
plt.ylabel("largeur pétal")
# plt.legend() 

plt.subplot(3,1,2)
plt.axis('equal')
plt.scatter(x[lab == "Iris-setosa"], y[lab == "Iris-setosa"], color='g', label='setosa')
plt.scatter(x[lab == "Iris-versicolor"], y[lab == "Iris-versicolor"], color='r', label='versicolor')
plt.scatter(x[lab == "Iris-virginica"], y[lab == "Iris-virginica"], color='b', label='virginica')
plt.scatter(longueur, largeur, color='k')
plt.xlabel("longueur pétal")
plt.ylabel("largeur pétal")
# plt.legend() 

plt.subplot(3,1,3)
plt.axis('equal')
plt.scatter(SL[lab == "Iris-setosa"], SW[lab == "Iris-setosa"], color='g', label='setosa')
plt.scatter(SL[lab == "Iris-versicolor"], SW[lab == "Iris-versicolor"], color='r', label='versicolor')
plt.scatter(SL[lab == "Iris-virginica"], SW[lab == "Iris-virginica"], color='b', label='virginica')
plt.scatter(l2, w2, color='k')
plt.xlabel("longueur sépal")
plt.ylabel("largeur sépal")
# plt.legend() 
plt.show()
#fin graphique

#algo knn
d=list(zip(x_scale+SL_scale,y_scale+SW_scale))
model = KNeighborsClassifier(n_neighbors=k)
model.fit(d,lab)
prediction= model.predict([[long_scale+slong,lar_scale+slar]])
print(long_scale+slong)
print(lar_scale+slar)
#fin algo knn 



#Affichage résultats
txt="Résultat : " 
if prediction[0]=="Iris-setosa":
  txt=txt+"setosa"
if prediction[0]=="Iris-versicolor":
  txt=txt+"versicolor"
if prediction[0]=="Iris-virginica":
  txt=txt+"virginica"

#fin affichage résultats


print(txt)	

