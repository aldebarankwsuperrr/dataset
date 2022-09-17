Laporan Proyek Mechine Learning - Putra Dwi Wira Gardha Yuniahans
-----------------------------------------------------------------

Domain Proyek
-----------------------------------------------------------------
Salah satu kebutuhan pokok manusia adalah tempat tinggal atau perumahan.
seiring meningkatnya jumlah manusia, akan mempengaruhi juga permintaan
konsumen terhadap perumahan juga meningkat[2]. Keadaan tersebut berdampak
pada konsumen apabila nanti akan memilih tempat tinggal mereka. Maka dari
itu proyek ini mencoba untuk menganalisis harga rumah berdasarkan parameter
tertentu. Proyek tersebut adalah proyek mechine learning dengan menggunakan
algoritma KNN serta Random Forest.

Business Understanding
-----------------------------------------------------------------
Berdasarkan uraian kondisi diatas, maka saya mengembangkan sebuah proyek 
sistem prediksi harga perumahan agar menjawab pertanyaan-pertanyaan 
berikut:
1. Dari berbagai banyak parameter atau fitur yang ada pada perumahan,
parameter apa yang berkorelasi atau berpengaruh terhadap harga perumahan
2. Berapakah harga pasaran rumah, dengan parameter atau fitur tertentu.

Agar menjawab pertanyaan tersebut, terdapat tujuan proyek atau sistem yang
saya buat. berikut merupakan tujuan dari proyek ini:
1. Mengetahui parameter atau fitur apa saja yang mempengaruhi atau 
berkorelasi dengan harga perumahan.
2. Membuat sebuah model mechine learning agar bisa memprediksi harga
perumahan berdasarkan parameter atau fitur yang ada secara akurat.

Solusi agar bisa mencapai goals atau tujuan dari proyek tersebut yaitu
saya menggunakan dataset dengan jumlah data 1010, serta algoritma yang 
saya gunakan pada proyek tersebut yaitu K-Nearest Neighbor dan 
Random Forest.

Data Understanding
-----------------------------------------------------------------
Pada proyek kali ini yaitu sistem prediksi harga perumahan berdasarkan
parameter atau fitur tertentu. Dataset untuk proyek tersebut yaitu menggunakan 
dataset DATA RUMAH.csv. <a href="https://drive.google.com/uc?export=download&id=1LKpakgkWUyDH43Q18ouX8RhblhLa9eWO
">download dataset</a>
Pada dataset tersebut berisikan jumlah data sebanyak 1010 data, serta terdapat
field sebanyak 8 field atau kolom. berikut merupakan uraian field pada
dataset tersebut:
1. NO         : No urutan data
2. NAMA RUMAH : Nama atau deskripsi Rumah
3. HARGA      : Harga Rumah
4. LB 	      : Merupakan deskripsi luas bangunan rumah tersebut
5. LT         : Merupakan deskripsi Luas Tanah dari rumah tersebut
6. KT         : Merupakan deskripsi banyak kamar tidur pada rumah tersebut
7. KM         : Merupakan deskripsi banyak kamar mandi pada rumah tersebut
8. GRS        : Merupakan deskripsi jumlah kendaraan yang bisa masuk garasi

Setelah mengetahui uraian field dari dataset tersebut selanjutnya yaitu mengetahui apakah terdapat data yang missing value.
Agar mengetahui hal tersebut terdapat function describe yang hasilnya seperti berikut:
![describe](https://user-images.githubusercontent.com/108270264/189850132-a87953be-f146-4818-a7f2-3a3d87382958.PNG)
<br>pada gambar tersebut terlihat bahwa tidak terdapat data yang missing value. ciri-ciri data yang missing value yaitu terdapat data yang berisi nilai 0
serta data tersebut tidak boleh terisi nilai 0. Selain mencari data yang missing value terdapat langkah selanjtnya yaitu mengetahu data yang outliers,
disini saya menggunakan bloxpot untuk mengetahui apakah data tersebut outliers. berikut merupakan visualisasi data outliers menggunakan boxplot:
![boxplot](https://user-images.githubusercontent.com/108270264/189850313-bbc3b991-86aa-4828-9c37-6d7fcc82ebf7.PNG)
<br>Setelah mengetahui data outliers selanjutnya terdapat teknik agar mengatasi data outliers tersebut. Pada proyek ini saya menggunakan teknik IQR atau juga seperti konsep kuartil pada statistika. Jadi bisa disimpulkan IQR = Kuartil 3 - Kuartil 1.

Data Preparation
-------------------------------------------------------------------
![cor](https://user-images.githubusercontent.com/108270264/189850383-c6887a4d-73e0-4ca5-bb03-cbc40c80d3c2.PNG)
<br>Pada proyek tersebut terdapat dua parameter atau fitur yang berkorelasi
atau berpengaruh terhadap harga rumah, yaitu Luas Bangunan dan Luas Tanah. dibuktikan dengan hasil corr yang mendekati angka 1 pada gambar diatas.
maka dari itu kedua fitur tersebut saya melakukan metode reduksi dimensi
dengan menggunakan metode PCA. berikut merupakan hasil dataset setelah melakukan proses reduksi dimensi<br>
![PCA](https://user-images.githubusercontent.com/108270264/189850499-898fde38-302a-4093-85f2-393a5dc3765b.PNG)
<br>Setelah melakukan reduksi dimensi selanjutnya
yaitu saya membagi dataset menjadi dua yaitu data train dan data test.
perbandingan dari pembagian data tersebut yaitu (80:20), karena pada saat
tahap Data Understanding yaitu menghilangkan data missing value dan data
outliers, jumlah dataset menjadi 695 data. maka dari itu saya membaginya
menjadi (80:20) agar data test bisa seimbang, untuk membagi dataset tersebut
saya menggunakan fungsi train_test_split dari library scikit learn. 
setelah melakukan pembagian dataset maka selanjutnya yaitu melakukan 
standarisasi data train saja, agar bisa menghidari kebocoran informasi 
pada data test atau data uji. Pada Standarisasi data saya menggunakan 
teknik StandardScaler dari Library ScikitLearn. berikut merupakan dataset setelah distandardisasi<br>
![standard](https://user-images.githubusercontent.com/108270264/189850574-ccb772cc-9e40-479f-97cf-1df21b01964a.PNG)

Modeling
--------------------------------------------------------------------
Setelah melakukan tahapn dari bussine understanding, data understanding, serta data preparation, maka selanjutnya beranjak pada tahapan modeling atau model development. Pada tahap ini saya menggunakan algoritma mechine learning untuk menyelesaikan permasalahan pada bussines understanding. algoritma yang akan dikembangkan untuk proyek ini terdapat algoritma K-Nearest Neighbor serta Algoritma Random Forest yang pada akhirnya akan di evaluasi dan dipilih salah satu algoritma yang paling baik. Pertama algoritma K-Nearest Neighbor, algoritma tersebut bisa dibilang salah satu algoritma yang relatif sederhana dibandingkan dengan algoritma mechine learning lainnya. Pada algoritma KNN konsep bekerja algoritma tersbut yaitu dengan menggunakan "Kesamaan Fitur" jadi KNN ini bekerja dengan membandingkan suatu jarak satu sampel ke sampel pelatihan lain dengan cara memilih nilai k tetangga terdekat. visualisasi dari algoritma ini yaitu sebagai berikut:<br>
![knn](https://user-images.githubusercontent.com/108270264/189883321-34f4d3a1-d5f4-4c45-b556-56a27a08cfc7.png)
<br>Jadi k atau tetangga yang sudah diteteapkan akan dicari titik dengan jarak rata-rata dari nilai k tersebut. Untuk menentukan titik dari hasil rata-rata jarak nilai k, algoritma KNN menggunakan perhitungan ukuran jarak yang defaultnya pada library scikitLearn yaitu Minkowski distance. Berbagai macam jenis perhitungan jarak yang sering kali dipakai yaitu Euclidean distance dan Manhattan distance. Untuk Euclidean merumuskan akar kuadrat dari jumlah selisih kuadrat antara titik a dan titik b sebagai berikut:
![rumusKNN](https://user-images.githubusercontent.com/108270264/189884703-2380fbae-4283-482c-98f8-9722dc0ceba5.jpeg)
<br>Sedangkan Minkowski distance sebagai generelasi dari kedua rumus diatas, untuk rumusnya sebagai berikut:<br>
![mink](https://user-images.githubusercontent.com/108270264/189885400-c5633c4b-174f-4766-bb83-fcfd43220a24.jpeg)
<br>kelebihan dari algoritma KNN yaitu algoritma yang mudah dipahami[1], tetapi
terdapat juga kekeruangan yaitu jika fitur dari dataset terlalu banyak,
maka algoritma KNN tidak bisa berjalan atau berfungsi secara maksimal. Pada algoritma KNN terdapat parameter yang digunakan, yaitu 'n_neighbors' merupakan jumlah tetangga untuk mengukur jarak antara titik. Pada proyek tersebut sebelum menggunakan GridSearchCV parameter 'n_neighbor' diset dengan nilai 20, sehingga hasil eror yang didapatkan yaitu train sebesar 915009062199820.25 serta test sebesar 2978275249568385.0.
<br><br>
Algoritma kedua yang saya gunakan yaitu Random Forest, algoritma tersebut merupakan salah satu dari banyak algoritma dari supervised learning. algoritma random forest juga bisa digunakan untuk menyelesaikan permasalahan klasifikasi dan regresi. salain termasuk algoritma supervised, random forest juga termasuk kategori algoritma ensemble yang berarti model prediksi yang terdiri dari beberapa model serta model terebut bekerja secara bersama-sama. Pada algoritma ensemble terdapat 2 teknik pendekatan yaitu bagging dan bossting. Bagging sendiri merupakan teknik melatih model dengan random sampel. Visualisasi dari algoritma ensemble teknik bagging sebagai berikut:
![ensemble](https://user-images.githubusercontent.com/108270264/189885934-a949138c-bcd3-420a-beb3-2d458643170d.png)
<br>Untuk kelebihan dari algoritma ini yaitu jika dihadapkan dengan dataset yang besar algoritma random forest akan bekerja secara maksimal. Selain kelebihan terdapat juga kekurangannya yaitu agar hasil akurasi dari algoritma ini tinggi maka diperlukan sumber daya dalam memproses komputasi yang artinya semakin banyak sumberdaya yang diperlukan juga semakin banyak waktu yang diperlukan. Paramater yang digunakan pada Algoritma tersebut yaitu :
1. Random Forest : -> n_estimator merupakan jumlah tree atau pohon
	           -> max_depth merupakan jumlah percabangan pada pohon
                   -> random_state merupakan pengontrol random number
                   -> n_jobs merupakan jumlah pekerjaan yang digunakan pada saar pararel.
Pada proyek kali ini sebelum menggunakan GridSearchCV parameter-parameter untuk algoritma Random Forest diset sebagai berikut:
![param](https://user-images.githubusercontent.com/108270264/190039331-560507c1-88a2-499f-a9d3-939f87bdaeee.PNG)

<br>Tahapan dalam implementasi kedua algoritma diatas yaitu :
1. K-Nearest Neighbor = model KNN merupakan salah satu library dari scikitLearn maka langkah pertama mengimport library scikitLearn terlebih dahulu, setelah itu membuat model dengan memanggil KNN yang didalamnya terdapat parameter "n_neighbors". langkah terakhir yaitu melatih model knn tersebut dengan data train dan data test.
2. Random Forest = model Random Forest merupakan salah satu library scikitLearn sama halnya seperti algoritma KNN, langkah pertama yaitu mengimport library scikitLearn terlebih dahulu, Selanjutnya membuat model dengan memanggil Random Forest yang didalamnya berisi parameter "n_estimator", "max_depth", "random_state", dan "n_jobs. Setelah membuat model, maka model bisa dilatih dengan data train data test.
<br> 

Pada tahap modeling selain menerapkan algoritma mechine learning yang dipakai terdapat juga proses pencarian parameter terbaik yang digunakan untuk algoritma yang sudah dijelaskan diatas. Proses tersebut merupakan proses Hyperparameter Tunning. berfungsi untuk mencari parameter dari model yang terbaik, sehingga kita tidak mencoba satu per satu model dengan parameter yang berbeda. Pada proses tersebut terdapat beberapa library yang dipakai, salah satunya dari library scikitLearn yaitu GridSearchCV. Pada proses ini juga mendapatkan hasil parameter terbaik dari setiap-setiap algoritma, Pertama pada algoritma KNN yang sebelumnya pada algoritma tersebut parameter 'n_neighbors' diset 20, dan setelah melakukan proses Hyperparameter Tunning ternyata parameter terbaik adalah nilai 'n_neighbors' menjadi 1. Hal yang sama pada algoritma Random Forest yang sebelumnya parameter-parameter diset seperti penjelasan sebelunya maka setelah proses Hyperparameter Tunning, parameter-parameternya menjadi seperti berikut :
![parame](https://user-images.githubusercontent.com/108270264/190040240-9c70fe21-eff1-4b68-9499-bb7fc7f3455f.PNG)
<br> untuk membuktikan apakah Hyperparameter Tunning menghasilkan nilai parameter yang terbaik maka dilihat dari nilai error masing-masing algoritma sebelum memakai Hyperparameter Tunning dan setelah memakainya. berikut merupakan perbedaan error tersebut:
<br>Gambar error sebelum dilakukan Hyperparameter Tunning<br>
![mse](https://user-images.githubusercontent.com/108270264/189876193-2e8d6d3f-a98d-4a38-9402-95fe26bf8541.PNG)
<br>Gambar error setelah melakukan Hyperparameter Tunning<br>
![hyper](https://user-images.githubusercontent.com/108270264/189861851-427ea22b-5a33-4872-b01d-e66f69657595.PNG)
<br>Bisa disimpulkan gambar diatas maka error yang dihasilkan setelah melakukan Hyperparameter Tunning lebih kecil dibanding sebelum melakukannya. Selanjutnya yaitu memilih dari kedua algoritma tersebut yang paling baik, disini saya memilih algoritma Random Forest, karena error yang dihasilkan lebih kecil dibandingkan dengan error yang dihasilkan oleh algoritma KNN.

Evaluasi
---------------------------------------------------------------------
Kasus pada proyek mechine learning saya merupakan kasus atau permasalahan
regresi serta pada proyek ini saya menggunakan metrik MSE yang digunakan untuk 
mengevaluasi seberapa baik model saya untuk memprediksi. Metrik MSE
merupakan kepanjangan dari Mean Squared Error serta metrik ini bekerja
dengan menghitung kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan persamaan sebagai berikut:<br>
![pers](https://user-images.githubusercontent.com/108270264/189878442-f24febb2-6513-4683-877b-f6ea90fb5f03.jpeg)

<br>Dengan menggunakan metrik MSE hasil evaluasi model dari proyek tersebut
yaitu:
1. Algoritma K-Nearest Neighbor :-> train : 915009062199820.25	
                                 -> test  : 2978275249568385.0
2. Algoritma Random Forest      :-> train : 3272183124384111.0	
                                 -> test  : 3779239237349212.5
![error](https://user-images.githubusercontent.com/108270264/189850708-6d5f0343-3c66-48f5-b63f-f92aa5e03c53.PNG)

<br>Berdasarkan gambar diatas serta error yamg dihasil kan masing2 algoritma, terlihat bahwa algoritma Random Forest memiliki eror yang paling kecil dibandingkan dengan error pada algoritma KNN. Maka model Random Forest lah yang akan dijadikan sebagai prediksi harga perumahan pada proyek ini. Selanjutnya yaitu mengetahui hasil prediksi dari kedua algoritma diatas, maka hasilnya bisa dilihat sebagai berikut
![hasil](https://user-images.githubusercontent.com/108270264/189878130-42f7d2a6-b14b-49d5-93ed-6fd7daf18586.PNG)


Kesimpulan
-------------------------------------------------------------------------
Kesimpulan dari proyek tersebut yaitu bahwasannya model bisa memprediksi namun akurasi yang didapatkan masih belum mencapai kesempurnaan meskipun error yang dihasilkan sudah kecil. Serta tujuan dari proyek ini sudah tercapai tetapi model bisa dikembangkan lagi hingga prediksi bisa mencapai akurasi yang tinggi dan bisa memprediksi secara akurat.

Refrensi
-------------------------------------------------------------------------
<div class="csl-entry">[1] Ernawati, S., &#38; Wati, R. (2018). <i>Penerapan Algoritma K-Nearest Neighbors Pada Analisis Sentimen Review Agen Travel</i>. <i>VI</i>(1). https://www.trustpilot.com/categories/tr</div>
<div class="csl-entry">[2] Widyasari, S., &#38; Fifilia, E. T. (n.d.). <i>ANALISIS PENGARUH PRODUK, HARGA, PROMOSI DAN LOKASI TERHADAP KEPUTUSAN PEMBELIAN RUMAH ( Studi pada Perumahan Graha Estetika Semarang )</i>.</div>
