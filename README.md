# Laporan Proyek Machine Learning

### Nama : Devi Indah Sari

### Nim : 211351044

### Kelas : Pagi B

## Domain Proyek
Web App ini merupakan pengimplementasian algorithma k-means clustering pada datasets Penguin untuk mengelompokkan penguin-penguin berdasarkan karakteristik fisiknya.

## Business Understanding
Web app ini dapat digunakan untuk memahami karakteristik penguin di Antartika. Dengan menggunakan teknik clustering, kita dapat membagi penguin menjadi beberapa kelompok berdasarkan karakteristik fisiknya. Tentunya hal ini dapat memudahkan ilmuwan dalam mengidenfikasi penguin tertentu termasuk pada cluster/species mana.

### Problem Statement
Dengan semakin maraknya isu global warming, penguin-penguin pada bagian antartica semakin terancam punah. Alangkah baiknya jika ilmuwan bisa dengan mudah mengkelompokkan penguin agar bisa memantau dan memastikan bahwa species tertentu terjaga.

### Goals
Menjaga species-species penguin dari kepunahan dengan melakukan pengelompokkan berdasarkan karakteristik fisiknya.

### Solution Statements

- Membuatkan web app yang bisa membuat mengelompokan secara otomatis.

## Data Understanding
Dataset ini mengandung data ukuran dari fisik penguin dalam satuan milimeter (mm). Dataset ini mengandung 344 baris data yang bisa kita kelompokkan dengan jumlah 5 kolom. <br>
(Clustering Penguins Species)[https://www.kaggle.com/datasets/youssefaboelwafa/clustering-penguins-species]

### Variabel-variabel pada Diabetes Prediction adalah sebagai berikut:

- culmen_length_mm : Menunjukkan panjang paruh penguin [float, 32.1-59.6]
- culmen_depth_mm : Menunjukkan kedalaman paruh penguin [float, 13.1-21.5]
- flipper_length_mm : Menunjukkan panjang sirip penguin [int, 172-231]
- body_mass_g : Menunjukkan berat badan penguin [int, 2700-6300]
- sex : Menunjukkan jenis kelamin penguin [MALE/FEMALE]

## Data Preparation
Seperti biasa pada tahap ini saya akan melakukan proses EDA dan pre-processing dalam satu tahap.

### Import Token Kaggle
Pertama kita harus memasukkan token kaggle yang telah kita unduh ke dalam google colab kita.
```python
from google.colab import files
files.upload()
```
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

### Download Dataset Dari Kaggle
Selanjutnya kita akan mengundah dataset dari kaggle.
```python
!kaggle datasets download -d youssefaboelwafa/clustering-penguins-species
```
```python
!unzip clustering-penguins-species.zip -d data
!ls data
```
### Import Library
```python
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pickle
```
### EDA, Data Discovery & Preprocessing
Mari mulai melakukan tahap Data Discovery
```python
df = pd.read_csv('data/penguins.csv')
```
Melihat 5 data paling atas.
```python
df.head()
```
Di atas bisa dilihat terdapat nilai NaN, tidak mana tidak kita inginkan dan akan kita hilangkan pada tahap pre-processing.
```python
df.info()
```
Tampaknya datasets yang kita pilih hanya memiliki 1 Dtype object, berarti kita hanya perlu mengganti nilai itu menjadi nilai numerical nanti.
```python
df.describe()
```
Jumlah data dari datasets ini adalah 342 baris dan 5 kolom. Kita akan periksa ulang jumlah baris datanya setelah dilakukan tahap pre-processing.
```python
df.isnull().sum()
```
Terdapat nilai null yang lumayan banyak ya! kita akan hilang di tahap yang akan datang nanti.
<br>
Okeh, ditahap ini kita akan melakukan explorasi data dan melakukan visualisasi serta analisis mengenai korelasi antar kolomnya.
```python
# Visualisasi kolom bill length
plt.figure(figsize=(10, 8))
sns.distplot(df["culmen_length_mm"])
plt.title("Distribusi panjang paruh penguin")
plt.show()
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/3b045188-bbbd-40ea-9153-75e904cb288a)<br>
kolom culmen_length_mm tampaknya sudah aman dan tidak ada skewness ya dari datanya. Kita akan lanjut dengan melihat kedalaman paruhnya.
```python
plt.figure(figsize=(10, 8))
sns.distplot(df["culmen_depth_mm"])
plt.title("Distribusi kedalaman paruh penguin")
plt.show()
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/a367e6e4-dc7a-43fc-ace4-bf340641e242)<br>
Sama seperti sebelumnya, kolom ini juga aman.
```python
sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm', hue = 'sex', data=df)
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/b7fec7f6-d385-4cdd-a20b-68f2caa6776b)<br>
hmmm....tampaknya data sex ini memiliki satu nilai unique yang tidak kita inginkan, yaitu ".", perlu dicatat tampilan diatas menunjukkan kolerasi antara kolom panjang paruh dan kedalaman paruh penguin. Bisa dilihat penguin gender male cenderung memiliki paruh yang lebih panjang dan lebih dalam dibandingkan paruh penguin female.
```python
plt.figure(figsize=(10, 8))
sns.boxplot(x="sex", y="body_mass_g", data=df)
plt.title("Perbandingan massa tubuh penguin berdasarkan gender")
plt.show()
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/0655730d-c95f-4513-89e5-b3b5b4ad1a49)<br>
Bisa dilihat dari box plot diatas, terdapat data yang tidak masuk akal karena nilainya yang terlalu tinggi dan terlalu rendah, ini yang dinamakan nilai outlier, karena didalam datasetsnya terdapat data outlier, kita harus menghilangkan itu terlebih dahulu agar visualisasi selanjutnya terlihat lebih rapih dan sesuai.
```python
print(df[df["flipper_length_mm"] > 4000])
print(df[df["flipper_length_mm"] < 0])
```
Nah, bisa kita lihat, terdapat 2 baris data yang memiliki nilai flipper tidak masuk akal, yaitu 5000 dan -132, maka harus kita hilangkan agar tidak terjadi keabnormalan untuk proses selanjutnya.
```python
df_no = df.drop([9, 14])
```
```python
plt.figure(figsize=(10, 8))
sns.scatterplot(x="culmen_length_mm", y="flipper_length_mm", hue='sex', data=df_no)
plt.title("Hubungan antara panjang paruh dan panjang sirip penguin berdasarkan sex")
plt.show()
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/db8957fe-ca37-4f25-a295-d712f4cd5ec5)<br>
Pada plot scatter diatas menunjukkan bahwa panjang paruh penguin dan flippernya memiliki hubungan based on gender penguin tersebut. <br>
Karena tadi kita menemukan data null pada datasets dan terdapat nilai '.' pada kolom sex, mari kita hilangkan nilai-nilai tersebut.
```python
df_clean = df_no.dropna()
df_clean = df_clean[df_clean['sex'] != '.']
df_clean
```
Lalu kita harus melihat ulang apakah datasetsnya masih memiliki nilai null dan kita harus reset_indexnya agar saat modeling tidak terdapat cluster yang nyasar (tidak sesuai dengan barisnya).
```python
df_clean.isnull().sum()
df_clean = df_clean.reset_index(drop=True)
df_clean.info()
```
Mari periksa ulang grafik perbandingan antara sex dan body_mass_g
```python
plt.figure(figsize=(10, 8))
sns.boxplot(x="sex", y="body_mass_g", data=df_clean)
plt.title("Perbandingan massa tubuh penguin berdasarkan gender")
plt.show()
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/4e58ffdb-2149-4287-9532-8bb1fe190254)
<br>
mulai sekarang data yang akan kita gunakan berada pada variable df_clean.
<br>
Karena kita akan menggunakan kolom sex, kita akan mengubahnya menjadi nilai 1 untuk male dan 0 untuk female.
```python
df_clean["sex"] = df_clean["sex"].replace({"MALE": 1, "FEMALE": 0})
```
```python
features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
data = df_clean[  ]
sse_scores = []
k_range = range(2, 11)  # Adjust k range based on your data

for k in k_range:
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(data)
  sse_scores.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, sse_scores, marker="o", label="SSE")
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE")
plt.title("SSE vs. Number of Clusters")
plt.legend()
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/86fa8cdc-bd8c-42ce-83fa-4003b0dfad42)<br>
Disini kita akan ambil 5 sebagai K nya karena perubahan SSE dari 5 ke 6 tidak begitu steep

## Modeling
Kita akan melakukan tahap modeling disini dengan menggunakan 5 jumlah cluster. Lalu memasukkan label hasil modeling ke dalam variable labels. Serta menunjukkan nilai tengah masing-masing cluster.
```python
kmeans = KMeans(n_clusters=5, random_state=42).fit(df_clean)
labels = pd.DataFrame(kmeans.labels_)
kmeans.cluster_centers_
```
Kita memasukkan nilai labels kedalam dataframe kita dengan membuat kolom baru bernama Labels
```python
df_clean['Labels'] = labels
```
Sip, sudah sesuai. selanjutnya adalah melakukan visualisasi clustering agar bisa melihat hasilnya dengan lebih jelas
### Visualisasi hasil algoritma
```python
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(df_clean['culmen_length_mm'],df_clean['body_mass_g'],c=labels,s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Culmen Length in mm')
ax.set_ylabel('Body Mass in g')
plt.colorbar(scatter)
```
![download](https://github.com/deviindah33/clustering-penguins/assets/149306739/369b086b-bc73-4454-8eab-5140aeb5f338)<br>
Diatas merupakan hasil clustering yang menunjukkan panjang paruh penguin dengan berat badannya.

## Deployment
(Web Clustering Penguin)[https://clustering-penguins-devi.streamlit.app/]
![image](https://github.com/deviindah33/clustering-penguins/assets/149306739/a4ebed1f-758c-49f0-9abe-1dba4c162af6)

