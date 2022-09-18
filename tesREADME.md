# Laporan Proyek Machine Learning - Fahrul Firmansyah

## Domain Proyek 

Susu merupakan salah satu bahan makanan tertua di dunia. Sejak 8.000 tahun sebelum masehi, susu sudah diolah untuk dijadikan bahan makanan. Hal ini tidaklah mengherankan, karena susu memiliki segudang manfaat jika kita mengonsumsinya, mulai dari kaya akan kalsium dan protein, membuat tidur lebih nyenyak, hingga mengurangi efek menstruasi pada wanita. Namun susu rawan mengalami kerusakan karena kadar air dan protein yang terkandung dalam susu sangat tinggi. Susu dengan kualitas yang tidak baik tentunya akan memberikan dampak negatif pada orang yang mengonsumsinya, hal ini dikarenakan pada susu yang mengalami kerusakan terdapat bakteri patogen akan menimbulkan penyakit bagi manusia. 

Oleh karena itu diperlukan adanya sebuah pengklasifikasian yang dapat membedakan antara susu yang baik dan layak dikonsumsi dengan susu yang mengalami kerusakan dan tidak layak dikonsumsi. Kualitas susu dapat diidentifikasi dari berbagai macam hal, mulai dari pH, temperatur, warna, dan lain-lain[1]. Pada proyek ini akan dibuat sebuah model machine learning yang dapat mendeteksi kualitas susu berdasarkan dataset susu yang dapat dikunjungi <a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"> disini </a>.

## Business Understanding
Selain ancaman bahaya kesehatan untuk tubuh, kesalahan dalam mengidentifikasi kualitas susu juga dapat membuat kekeliruan dalam menentukan harga jual susu. Susu dengan kualitas tinggi tentu memiliki harga jual lebih tinggi dibandingkan dengan susu dengan kualitas buruk. Dari pernyataan tersebut, dapat ditarik kesimpulan bahwa permasalahan utama dapat dinyatakan dengan sebuah pertanyaan berikut:
- Bagaimana cara membuat model machine learning dengan akurasi tinggi dalam mengidentifikasi kualitas susu berdasarkan fitur-fitur tertentu?

Dalam menyelesaikan permasalahan tersebut, berikut beberapa solusi yang akan dilakukan pada proyek ini:
- Dibuatlah sebuah <i>predictive model</i> dengan tujuan unntuk mengetahui kualitas susu berdasarkan fitur-fitur tertentu dengan menggunakan <a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"> dataset </a> dengan jumlah sampel 1059 data. Dalam membuat model machine learning, akan digunakan 3 model berbeda dengan menerapkan hyperparameter tuning pada setiap modelnya, kemudian akurasi pada tiap model akan diukur menggunakan  metode mean square error, model dengan error paling rendah akan diambil sebagai model utama.
  
## Data Understanding
<a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality">Dataset</a> yang digunakan pada proyek ini diambil dengan manual dengan metode observasi. Target yang digunakan adalah kolom 'Grade' dengan tiga nilai yaitu low, medium, dan High.

Variabel-variabel pada dataset susu adalah sebagai berikut :
- Taste       : rasa dari susu, variabel berisi dua nilai yaitu 1 dan 0
- Odor        : bau dari susu, variabel berisi dua nilai yaitu 1 dan 0
- Fat         : lemak dari susu, variabel berisi dua nilai yaitu 1 dan 0
- Turbidity   : kekeruhan dari susu, variabel berisi dua nilai yaitu 1 dan 0
- pH          : pH dari susu, variabel berisi nilai variatif diambil berdasarkan sampel
- Temperature : Temperatur dari susu, variabel berisi nilai variatif diambil berdasarkan sampel
- Colour      : warna dari susu, variabel berisi nilai variatif diambil berdasarkan sampel. 

Untuk penjelasan lebih rinci pada dataset dapat dilihat pada tabel berikut

|   Column   | Non-Null Count |  Dtype  |
|:----------:|:--------------:|:-------:|
|     PH     |  1059 non-null | float64 |
| Temprature |  1059 non-null |  int64  |
|    Taste   |  1059 non-null |  int64  |
|    Odor    |  1059 non-null |  int64  |
|     Fat    |  1059 non-null |  int64  |
|  Turbidity |  1059 non-null |  int64  |
|   Colour   |  1059 non-null |  int64  |
|    Grade   |  1059 non-null |  object  |

Dari gambar diatas dapat dilihat bahwa dataset memiliki 7 kolom dengan tipe number baik int maupun float, dan satu kolom dengan tipe object yaitu Grade, Grade merupakan label pada dataset ini.

Setelah memahami dataset, tidak lupa juga melakukan pemeriksaan <i>missing value</i>. Dalam memeriksa <i>missing value</i> pada dataset, kita dapat menggunakan fungsi <i>isnull()</i>. Berikut tabel yang berisi jumlah <i>missing value</i> pada tiap fitur

|index|jumlah|
|---|---|
|pH|0|
|Temprature|0|
|Taste|0|
|Odor|0|
|Fat |0|
|Turbidity|0|
|Colour|0|
|Grade|0|

Dari tabel diatas dapat dilihat bahwa tidak terdapat <i>missing value</i> pada dataset.
  
Syarat dari dataset yang baik untuk digunakan dalam pembuatan model machine learning salah satunya haruslah seimbang. Salah satu cara memeriksa apakah dataset kita seimbang atau tidak adalah dengan melakukan visualisasi. Berikut visualiasi dari dataset susu yang akan digunakan pada pembuatan model machine learning proyek ini.

| Grade  | Jumlah Sampel | Presentase |  
|--------|---------------|------------|
| Low    |      429      |    40.5    |   
| Medium |      374      |    35.3    |   
| High   |      256      |    24.2    |  

Pada gambar diatas dapat dilihat bahwa dataset memiliki persebaran data yang cukup seimbang pada setiap nilai target, sehingga dataset susu dapat digunakan. 

Selanjutnya pemeriksaan fitur numerik, berikut histogram dari masing-masing fitur pada dataset susu.
<br>![persebaran_data](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/persebaran_data.png)<br>

dari gambar tersebut dapat ditarik beberapa hal:
- Pada fitur pH, dapat dilihat bahwa beberapa data terpusat pada antara pH 6 hingga pH 7.
- Pada fitur pH, terdapat sebagian kecil data memiliki nilai diatas 9. Hal ini dapat kita indikasikan sebagai outliers.
- Pada fitur Temprature, banyak data memiliki temprature dibawah 50, dan terdapat beberapa data memiliki terletak jauh dari data lain yaitu dengan nilai temprature 90, hal itu dapat kita indikasikan sebagai outliers.
- Fitur selanjutnya yang dapat diamati adalah fitur Colour, pada fitur colour data memiliki nilai yang variatif, namun dapat diamati bahwa terdapat sebuah data berada pada nilai colour 240, hal itu dapat diindikasikan sebagai outliers.

Selanjutnya adalah tahap memerikas <i>outliers</i>. Untuk memeriksa <i>outliers</i> pada dataset kita dapat melakukan beberapa hal. Salah satu metode yang dapat dilakukan dalam menangani <i>outliers</i> yaitu IQR. Metode IQR merupakan metode penanganan <i>outliers</i> dengan menerapkan batas atas dan batas bawah, kemudian data yang berada diluar batas atas dan batas bawah akan dianggap sebagai <i>outliers</i>. Karena hanya sebagian fitur pada dataset yang memiliki nilai yang variatif, maka pemeriksaan <i>outliers</i> hanya dilakukan pada fitur tersebut yaitu pH, Colour, dan Temprature. Berikut gambar boxplot IQR pada ketiga fitur:

![ph](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/ph.png)
![colour](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/colour.png)
![temprature](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/download.png)
<br>
Dari ketiga gambar diatas dapat diketahui bahwa terdapat beberapa <i>outliers</i> pada dataset, sehingg perlu dilakukan penanganan, dengan menerapkan <i>Bloxpot IQR</i> maka sampel pada dataset tersisa 648 sampel.

## Data Preparation

Untuk membuat dataset lebih mudah dipahami oleh model, maka dataset harus disiapkan sedemikian rupa. Ada beberapa metode yang dapat digunakan dalam tahap data preparation, metode yang akan digunakan dalam data preparation dalam proyek ini yaitu:
- merubah niai pada label menjadi bentuk numerik 
  hal ini karena label memiliki tiga nilai yaitu low, medium, dan high. Hal ini dilakukan agar model dapat dengan mudah melakukan prediksi. Fungsi yang digunakan pada tahap ini adalah fungsi <i>replace</i> dari library pandas
- Pembagian dataset dengan menggunakan train_test_split
  train_test_split merupakan fungsi dari library scikit-learn yang memiliki fitur untuk membagi data kedalam train data dan test data. Pembagian data ini diperlukan     agar model dapat diukur akurasinya. Dengan proporsi pembagian antara data latih dan data tes sebesar 70 : 30.
- Standarisasi
  Standarisasi merupakan kegiatan merubah nilai pada dataset sehigga pada range tertentu. Range yang besar antar nilai pada dataset dapat menyulitkan model dalam mempelajari pada dataset. Fitur-fitur yang akan di-standarisasi yaitu pH, Temprature, dan Colour. Untuk fitur-fitur lain tidak dilakukan standirisasi karena nilainya 1 dan 0. Fungsi yang akan digunakan pada tahap ini adalah <i>StandardSCaler</i> dari library scikit-learn.
 
## Modeling
Proyek ini memiliki fokus permasalahan pada indentifikasi kualitas susu, permasalah tersebut dalam <i>machine learning</i> digolongkan pada permasalahan klasifikasi. Beberapa algoritma yang dapat dipakai yaitu KNN, Random Forest, dan AdaBoost. Selain itu untuk memaksimalkan hasil yang didapat, maka akan dilakukan <i>hyperparameter tuning</i> pada setiap model yang akan diuji.

### Pemodelan KNN
KNN merupakan algoritma <i>machine learning</i> yang bekerja dengan mencari "kesamaan fitur" dalam melakukan prediksi. berikut penjelasan kelebihan dan kekurangan dari KNN:

#### Kelebihan :
- Memiliki bentuk yang sederhana
- Mudah diimplementasikan

#### Kekurangan :
- Tidak cocok untuk dataset besar

#### Parameter :
- n_neighbors = merupakan jumlah "tetangga" terdekat yang akan diklasifikasikan dalam satu kelompok. Diantara (5, 10, 15) menggunakan metode GridSearch didapat bahwa nilai terbaik untuk parameter ini adalah 5.

### Pemodelan Random Forest
Random merupakan algoritma <i>machine learning</i> yang bekerja dengan menggabungkan beberapa DecisionTree dalam melakukan prediksi.  Berikut penjelasan kelebihan dan kekurangan dari Random Forest:

#### Kelebihan :
- Ampuh dalam mengatasi noise 
- Mampu mengatasi missing value
- Dapat digunakan dalam data besar

#### Kekurangan :
- Membutuhkan tuning parameter yang tepat agar akurasi yang diharapkan dapat tercapai

#### Parameter :
- n_estimator = jumlah decision tree. Diantara (5, 10, 15) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 15.
- max_depth = panjang decisoing tree dalam membentuk node. Diantara (10, 16, 20) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 10.
- random_state = untuk mengatur random number generator. Diantara (45, 55, 65) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 55.
- n_jobs = jumlah pekerjaan yang berjalan dengan paralel. Diantara (1, 2, 3) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1.

### Pemodelan AdaBoost 
AdaBoost merupakan salah satu algoritma <i>machine learning</i> yang bekerja dengan cara "memperbaiki diri". Berikut penjelasan kelebihan dan kekurangan dari AdaBoost :

#### Kelebihan :
- Mudah diimplementasikan
- Relatif cepat dalam pengujian

#### Kekurangan :
- Membutuhkan dataset dengan akurasi tinggi

#### Parameter :
- learning_rate = weight pada setiap regressor tiap iterasi. Diantara (0.5, 0.05, 0.005) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0.5.
- random_state = untuk mengatur random number generator. Diantara (5, 55, 555) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 5.

Setelah parameter setiap model didapat, maka akan dilakukan pelatihan ketiga model menggunakan parameter yang didapat menggunakan Grid Search.

## Evaluation
Pada tahap evaluasi akan digunakan <i>mean square error</i> untuk menghitung error prediksi kualitas susu dari model dengan kualitas susu sesuai data. <i>Mean square error</i> bekerja dengan menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Berikut formula dari mean square error

<br>![formula](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/formula.jpeg)<br>

Keterangan : <br>
N = jumlah dataset <br>
yi = nilai sebenarnya <br>
y_pred = nilai prediksi <br>

Karena sebelumnya kita melakukan standarisasi pada data train, maka pada tahap evaluasi kita juga harus melakukan standariasi pada data test dalam menguji model. Berikut hasil evaluasi yang didapat dari ketiga model yang telah dilatih

|index|train|test|
|---|---|---|
|KNN|0.000008|0.000023|
|RF|0.000002|0.00003|
|Boosting|0.000059|0.00011|

Berikut visualisasi dari tabel diatas

<br>![evaluation](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/e.png)<br>

Diagram batang diatas merupakan jumlah error yang didapat setiap model pada saat diuji dengan data test. Dari diagram tersebut dapat dilihat bahwa KNN memiliki error paling rendah pada test set walaupun pada training set bukan error yang terendah. Maka KNN akan digunkan pada proyek ini dalam menentukan kualitas dari susu berdasarkan fitur pH, Temprature, dan lain-lain. Berikut sebagian kecil cuplikan dari test set

|index|y\_true|prediksi\_KNN|prediksi\_RF|prediksi\_Boosting|
|---|---|---|---|---|
|141|3|3\.0|3\.0|2\.9|
|645|3|3\.0|3\.0|2\.6|
|85|3|3\.0|3\.0|2\.9|
|806|2|2\.0|2\.0|2\.3|
|310|3|3\.0|3\.0|2\.9|
|552|3|2\.6|3\.0|2\.4|
|348|2|2\.0|2\.0|2\.0|
|522|2|2\.0|2\.0|2\.5|
|889|3|3\.0|3\.0|2\.9|

## Kesimpulan
Dengan menerapkan beberapa metode, dari ketiga model yang telah diuji, KNN
merupakan model yang memiliki error paling rendah dan mampu memprediksi kualitas dari susu dengan tepat. Tujuan dari proyek dalam mengatasi permasalahan yang telah disebutkan telah dapat tercapai.

## Referensi
<div class="csl-entry">[1] Ernawati, S., &#38; Setyorini. D.A. DKK. (2018). <i>ualitas dan Kuantitas Produksi Susu Sapi di Kemitraan PT. Greenfields Indonesia Ditinjau dari Ketinggian Tempat</i>. https://ejournal.unib.ac.id/index.php/jspi/article/view/12295</div>
