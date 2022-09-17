# Laporan Proyek Machine Learning - Fahrul Firmansyah

## Domain Proyek 

Susu merupakan salah satu bahan makanan tertua di dunia. Sejak 8.000 tahun sebelum masehi, susu sudah diolah untuk dijadikan bahan makanan. Hal ini tidaklah mengherankan, karena susu memiliki segudang manfaat jika kita mengonsumsinya, mulai dari kaya akan kalsium dan protein, membuat tidur lebih nyenyak, hingga mengurangi efek menstruasi pada wanita. Namun susu rawan mengalami kerusakan karena kadar air dan protein yang terkandung dalam susu sangat tinggi. Susu dengan kualitas yang tidak baik tentunya akan memberikan dampak negatif pada orang yang mengonsumsinya, hal ini dikarenakan pada susu yang mengalami kerusakan terdapat bakteri patogen akan menimbulkan penyakit bagi manusia. <br><br>

Oleh karena itu diperlukan adanya sebuah pengklasifikasian yang dapat membedakan antara susu yang baik dan layak dikonsumsi dengan susu yang mengalami kerusakan dan tidak layak dikonsumsi. Kualitas susu dapat diidentifikasi dari berbagai macam hal, mulai dari pH, temperatur, warna, dan lain-lain. Pada proyek ini akan dibuat sebuah model machine learning yang dapat mendeteksi kualitas susu berdasarkan dataset susu yang dapat dikunjungi <a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"> disini </a>
<br> 
Refrensi  : [Kualitas dan Kuantitas Produksi Susu Sapi di Kemitraan PT. Greenfields Indonesia Ditinjau dari Ketinggian Tempat](https://ejournal.unib.ac.id/index.php/jspi/article/view/12295)

## Business Understanding
Selain ancaman bahaya kesehatan untuk tubuh, kesalahan dalam mengidentifikasi kualitas susu juga dapat membuat kekeliruan dalam menentukan harga jual susu. Susu dengan kualitas tinggi tentu memiliki harga jual lebih tinggi dibandingkan dengan susu dengan kualitas buruk. Dari pernyataan tersebut, dapat ditarik kesimpulan bahwa permasalahan utama dapat dinyatakan dengan sebuah pertanyaan-pertanyaan berikut:
- Apa kualitas sebuah susu berdasarkan fitur-fitur tertentu.
- Bagaimana cara membuat model machine learning dengan akurasi tinggi dalam mengidentifikasi kualitas susu berdasarkan fitur-fitur tertentu.

Dalam menyelesaikan permasalahan tersebut, dibuatlah sebuah predictive model dengan tujuan unntuk mengetahui kualitas susu berdasarkan fitur-fitur tertentu dengan menggunakan <a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"> dataset </a> dengan jumlah sampel 1059 data. Dalam membuat model machine learning, akan digunakan 3 model berbeda dengan menerapkan hyperparameter tuning pada setiap modelnya, kemudian akurasi pada tiap model akan diukur menggunakan  metode mean square error, model dengan error paling rendah akan diambil sebagai model utama.

## Data Understanding
<a href="https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"> Dataset </a>

