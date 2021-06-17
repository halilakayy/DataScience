#!/usr/bin/env python
# coding: utf-8

# <b>Kütüphanelerin Yüklenmesi</b>

# In[1]:


"""
Kütüphaneler
"""
#Kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# <b>Verilerin Yüklenmesi</b>

# In[2]:


#Verinin içeri alınması
veriler=pd.read_csv('veriler.csv')
ulke=veriler[['ulke']]
boy=veriler[['boy']]
kilo=veriler[['kilo']]
yas=veriler[['yas']]
cinsiyet=veriler[['cinsiyet']]


print(veriler)
print(type(ulke))


# <b>Eksik Verilerin Tamamlanması</b>

# In[3]:


from sklearn.impute import SimpleImputer
veriler=pd.read_csv('eksikveriler.csv')
#print(eksikVeriler)


imputer=SimpleImputer(missing_values=np.nan, strategy='mean')

Yas=veriler.iloc[:,1:4].values
#print(Yas)

imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])

print(Yas)


# <b>Kategorik Verilerin Dönüşümü</b>

# In[4]:


ulke=veriler.iloc[:,0:1].values
cinsiyet=veriler.iloc[:,4:5].values
#print(ulke)
#print(cinsiyet)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
#print(ulke)

ulke=ohe.fit_transform(ulke).toarray()
#print(ulke)

cinsiyet[:,0]=le.fit_transform(veriler.iloc[:,4])
#print(cinsiyet)
cinsiyet=ohe.fit_transform(cinsiyet).toarray()
#print(cinsiyet)

sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr','us'])
#print(sonuc)
sonuc2=pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
#print(sonuc2)
cinsiyet=veriler.iloc[:,-1].values
#print(cinsiyet)
sonuc3=pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
#print(sonuc3)

s=pd.concat([sonuc, sonuc2], axis=1)
#print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)


# <b>Öğrenme ve Test(Trainging and Test)</b>

# In[5]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(s, sonuc3, test_size=0.33, random_state=0)

print("{} \n\n {}".format(x_train, x_test)) #birinci kolon öğrenme ve test görünümü yazdırma

print("{} \n\n {}".format(y_train, y_test))#ikinci kolon öğrenme ve test görünümü yazdırma


# In[6]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

print("{} \n\n {}".format(X_train, X_test))


# In[ ]:




