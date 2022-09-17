# Laporan Proyek Machine Learning - Fahrul Firmansyah

## Domain Proyek 

Susu merupakan salah satu bahan makanan tertua di dunia. Sejak 8.000 tahun sebelum masehi, susu sudah diolah untuk dijadikan bahan makanan. Hal ini tidaklah mengherankan, karena susu memiliki segudang manfaat jika kita mengonsumsinya, mulai dari kaya akan kalsium dan protein, membuat tidur lebih nyenyak, hingga mengurangi efek menstruasi pada wanita. Namun susu rawan mengalami kerusakan karena kadar air dan protein yang terkandung dalam susu sangat tinggi. Susu dengan kualitas yang tidak baik tentunya akan memberikan dampak negatif pada orang yang mengonsumsinya, hal ini dikarenakan pada susu yang mengalami kerusakan terdapat bakteri patogen akan menimbulkan penyakit bagi manusia. 

Oleh karena itu diperlukan adanya sebuah pengklasifikasian yang dapat membedakan antara susu yang baik dan layak dikonsumsi dengan susu yang mengalami kerusakan dan tidak layak dikonsumsi. Kualitas susu dapat diidentifikasi dari berbagai macam hal, mulai dari pH, temperatur, warna, dan lain-lain. Pada proyek ini akan dibuat sebuah model machine learning yang dapat mendeteksi kualitas susu berdasarkan dataset susu yang dapat dikunjungi <a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"> disini </a>.

Refrensi  : [Kualitas dan Kuantitas Produksi Susu Sapi di Kemitraan PT. Greenfields Indonesia Ditinjau dari Ketinggian Tempat](https://ejournal.unib.ac.id/index.php/jspi/article/view/12295)

## Business Understanding
Selain ancaman bahaya kesehatan untuk tubuh, kesalahan dalam mengidentifikasi kualitas susu juga dapat membuat kekeliruan dalam menentukan harga jual susu. Susu dengan kualitas tinggi tentu memiliki harga jual lebih tinggi dibandingkan dengan susu dengan kualitas buruk. Dari pernyataan tersebut, dapat ditarik kesimpulan bahwa permasalahan utama dapat dinyatakan dengan sebuah pertanyaan-pertanyaan berikut:
- Apa kualitas sebuah susu berdasarkan fitur-fitur tertentu?
- Bagaimana cara membuat model machine learning dengan akurasi tinggi dalam mengidentifikasi kualitas susu berdasarkan fitur-fitur tertentu?

Dalam menyelesaikan permasalahan tersebut, dibuatlah sebuah predictive model dengan tujuan unntuk mengetahui kualitas susu berdasarkan fitur-fitur tertentu dengan menggunakan <a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"> dataset </a> dengan jumlah sampel 1059 data. Dalam membuat model machine learning, akan digunakan 3 model berbeda dengan menerapkan hyperparameter tuning pada setiap modelnya, kemudian akurasi pada tiap model akan diukur menggunakan  metode mean square error, model dengan error paling rendah akan diambil sebagai model utama.

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

untuk penjelasan lebih rinci pada dataset dapat dilihat pada gambar berikut

<br>![info](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/info.jpg)<br>
Dari gambar diatas dapat dilihat bahwa dataset memiliki 7 kolom dengan tipe number baik int maupun float, dan satu kolom dengan tipe object yaitu Grade, Grade merupakan label pada dataset ini.

Syarat dari dataset yang baik untuk digunakan dalam pembuatan model machine learning salah satunya haruslah seimbang. Salah satu cara memeriksa apakah dataset kita seimbang atau tidak adalah dengan melakukan visualisasi. Berikut visualiasi dari dataset susu yang akan digunakan pada pembuatan model machine learning proyek ini.

<br>![grade](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/grade.png)<br>

Pada gambar diatas dapat dilihat bahwa dataset memiliki persebaran data yang cukup seimbang pada setiap nilai target, sehingga dataset susu dapat digunakan. 

Selanjutnya pemeriksaan fitur numerik, berikut histogram dari masing-masing fitur pada dataset susu.
<br>![persebaran_data](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/persebaran_data.png)<br>

dari gambar tersebut dapat ditarik beberapa hal:
- Pada fitur pH, dapat dilihat bahwa bebrapa data terpusat pada antara pH 6 hingga pH 7.
- Pada fitur pH, terdapat sebagian kecil data memiliki nilai diatas 9. Hal ini dapat kita indikasikan sebagai outliers.
- Pada fitur Temprature, banyak data memiliki temprature dibawah 50, dan terdapat beberapa data memiliki terletak jauh dari data lain yaitu dengan nilai temprature 90, hal itu dapat kita indikasikan sebagai outliers.
- Fitur selanjutnya yang dapat diamati adalah fitur Colour, pada fitur colour data memiliki nilai yang variatif, namun dapat diamati bahwa terdapat sebuah data berada pada nilai colour 240, hal itu dapat diindikasikan sebagai outliers.
- variasi nilai pada fitur-fitur dataset tidak terdapat nilai yang "rancu", sehingga kita dapat simpulkan tidak terdapat missing value pada dataset.

Untuk menangani outliers pada dataset kita dapat melakukan beberapa hal. Jika data yang outliers memiliki presentase kecil terhadap data, maka kita dapat menghilangkan saja data tersebut, agar model dapat memiliki akurasi yang tinggi. Seperti yang kita ketahui pada pemeriksaan fitur numerik, terdapat beberapa fitur memiliki outliers dengan presentase terhadap dataset sangat kecil, sehingga akan dilakukan penghapusan pada data tersebut.

## Data Preparation

Untuk membuat dataset lebih mudah dipahami oleh model, maka dataset harus disiapkan sedimikian rupa. Ada beberapa metode yang dapat digunakan dalam tahap data preparation, metode yang akan digunakan dalam data preparation dalam proyek ini yaitu:
- encoding pada label 
  hal ini karena label memiliki tiga nilai yaitu low, medium, dan high. Hal ini dilakukan agar model dapat dengan mudah melakukan prediksi.
- Pembagian dataset dengan menggunakan train_test_split
  train_test_split merupakan fungsi dari library scikit-learn yang memiliki fitur untuk membagi data kedalam train data dan test data. Pembagian data ini diperlukan     agar model dapat diukur akurasinya.
- Standarisasi
  Standarisasi merupakan kegiatan merubah nilai pada dataset sehigga pada range tertentu. Range yang besar antar nilai pada dataset dapat menyulitkan model dalam mempelajari pada dataset. Fitur-fitur yang akan di-standarisasi yaitu pH, Temprature, dan Colour. Untuk fitur-fitur lain tidak dilakukan standirisasi karena nilainya 1 dan 0.
 
 ## Modeling
 Proyek ini memiliki fokus permasalahan pada indentifikasi kualitas susu, permasalah tersebut dalam machine learning digolongkan pada permasalah klasifikasi. Beberapa algoritma yang dapat dipakai yaitu KNN, Random Forest, dan AdaBoost. Selain itu untuk memaksimalkan hasil yang didapat, maka akan dilakukan hyperparameter tuning pada setiap model yang akan diuji.

### Pemodelan KNN
KNN merupakan algoritma machine learning yang bekerja dengan mencari "kesamaan fitur" dalam melakukan prediksi. berikut penjelasan kelebihan dan kekurangan dari KNN:

#### Kelebihan :
- memiliki bentuk yang sederhana
- mudah diimplementasikan

#### Kekurangan :
- tidak cocok untuk dataset besar

#### Parameter :
- n_neighbors = merupakan jumlah "tetangga" terdekat yang akan diklasifikasikan dalam satu kelompok. Diantara (5, 10, 15) menggunakan metode GridSearch didapat bahwa nilai terbaik untuk parameter ini adalah 5.

### Pemodelan Random Forest
Random merupakan algoritma machine learning yang bekerja dengan menggabungkan beberapa Decision Tree dalam melakukan prediksi.  berikut penjelasan kelebihan dan kekurangan dari Random Forest:

#### Kelebihan :
- ampuh dalam mengatasi noise 
- mampu mengatasi missing value
- dapat digunakan dalam data besar

#### Kekurangan :
- membutuhkan tuning parameter yang tepat agar akurasi yang diharapkan dapat tercapai

#### Parameter :
- n_estimator = jumlah decision tree. Diantara (5, 10, 15) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 15.
- max_depth = panjang decisoing tree dalam membentuk node. Diantara (10, 16, 20) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 10.
- random_state = untuk mengatur random number generator. Diantara (45, 55, 65) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 55.
- n_jobs = jumlah pekerjaan yang berjalan dengan paralel. Diantara (1, 2, 3) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1.

### Pemodelan AdaBoost 
AdaBoost merupakan salah satu algoritma machine learning yang bekerja dengan cara "memperbaiki diri". Berikut penjelasan kelebihan dan kekurangan dari AdaBoost :

#### Kelebihan :
- mudah diimplementasikan
- relatif cepat dalam pengujian

#### Kekurangan :
- membutuhkan dataset dengan akurasi tinggi

#### Parameter :
- learning_rate = weight pada setiap regressor tiap iterasi. Diantara (0.5, 0.05, 0.005) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0.5.
- random_state = untuk mengatur random number generator. Diantara (5, 55, 555) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 5.

Setelah parameter setiap model didapat, maka akan dilakukan pelatihan ketiga model menggunakan parameter yang didapat menggunakan Grid Search.

## Evaluation
Pada tahap evaluasi akan digunakan mean squre error untuk menghitung error prediksi kualitas susu dari model dengan kualitas susu sesuai data. Mean square error bekerja dengan menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Berikut formula dari mean square error

<br>![formula](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/formula.jpeg)<br>

Keterangan :
N = jumlah dataset <br>
yi = nilai sebenarnya <br>
y_pred = nilai prediksi <br>

Karena sebelumnya kita melakukan standarisasi pada data train, maka pada tahap evaluasi kita juga harus melakukan standariasi pada data test dalam menguji model. Berikut hasil evaluasi yang didapat dari ketiga model yang telah dilatih

<br>![evaluation](https://raw.githubusercontent.com/aldebarankwsuperrr/dataset/main/eval.png)<br>


