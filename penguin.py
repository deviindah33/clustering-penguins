import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('penguins.csv')
df2 = pd.read_csv('cleaned_data.csv')
features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]

st.header('Segmentation Data Penguin Dataset menggunakan K-Means Clustering') 
#deskripsikan proyek ini
st.write('Proyek ini bertujuan untuk melakukan segmentasi data penguin dataset menggunakan metode K-Means Clustering. Data yang digunakan adalah data penguin dataset yang diambil dari Kaggle. Data ini berisi tentang jenis kelamin, dan ukuran fisiknya. Data yang digunakan adalah data yang telah dibersihkan dari data yang kosong.')
st.subheader('Data Asli')
st.write(df)

#add x
x = df2.iloc[:, [0,3]].values

st.subheader('Menentukan Jumlah Kluster yang Optimal dengan Metode Elbow')
k = []
for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(x)
    k.append(kmeans.inertia_)

plt.figure(figsize=(20,10))
plt.plot(range(1,11), k, marker='*')
plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
elbow = st.pyplot()
st.write('Dari grafik di atas, dapat dilihat bahwa titik siku berada pada nilai 2. Oleh karena itu, jumlah kluster yang optimal adalah 2.')

st.sidebar.subheader('Tentukan Jumlah Kluster')
cluster = st.sidebar.slider('Silahkan Tentukan Jumlah Kluster', 1, 10) 

def k_means(n):
    kmeans = KMeans(n_clusters=n, random_state=42).fit(x)
    df2['Labels'] = kmeans.labels_

    plt.figure(figsize=(12,8))
    sns.scatterplot(x='culmen_length_mm', y='body_mass_g', hue='Labels', data=df2, palette='viridis', marker='o', s=100)
    
    for label in df2['Labels']:
        plt.annotate(label, 
        (df2[df2['Labels']==label]['culmen_length_mm'].mean(),
        df2[df2['Labels']==label]['body_mass_g'].mean()),
        horizontalalignment='center',
        verticalalignment='center',
        size=20, weight='bold',
        color='Red')

    st.header('Hasil Yang Diperoleh dari Klustering')
    st.subheader('Hasil akhir data yang telah dikelompokkan')
    st.pyplot()
    st.write('Dari hasil klustering, dapat dilihat bahwa terdapat 2 kelompok penguin. Kelompok pertama adalah penguin yang memiliki ukuran fisik yang lebih besar, sedangkan kelompok kedua adalah penguin yang memiliki ukuran fisik yang lebih kecil.')
    st.write(df2)

k_means(cluster)
#buatkan kesimpulan dari hasil klustering yang telah dilakukan
st.subheader('Kesimpulan')
st.write('Dari hasil yang diperoleh, dapat kita simpulkan bahwa kemungkinan besar penguin yang memiliki ukuran fisik yang lebih besar adalah penguin jantan, sedangkan penguin yang memiliki ukuran fisik yang lebih kecil adalah penguin betina.')
st.write('Serta, dapat kita simpulkan juga bahwa kriteria fisik bisa menjadi salah satu faktor untuk menentukan species maupun jenis kelamin dari penguin.')