# tubes-ai-kelompok-11

Simulasi berbasis HTML/JavaScript (satu file, langsung dibuka di browser, tanpa server) yang memperagakan cara kerja algoritma pencarian UCS (Uniform Cost Search) dan A* lewat skenario kejar-kejaran: drone (pengejar, lingkaran merah) mencoba menemukan dan mengejar tank (pemain, kotak hijau) di atas medan perang bergrid 16 x 22 sel.

Cara menjalankan

Cukup buka file .html-nya di browser mana pun (Chrome/Edge/Firefox). Tidak perlu instalasi, build step, atau koneksi internet.

Alur permainan
Pemindaian awal — begitu simulasi dimulai, drone belum tahu posisi tank. Ia melakukan UCS dari titik awalnya, divisualisasikan sebagai gelombang warna (biru → ungu, dari biaya rendah ke tinggi) sampai "menemukan" posisi awal tank.
Mode menyisir  — setelah posisi awal tank diketahui, drone kembali "lupa" dan menyisir peta memakai pola tetap sambil menjaga radius pandang. Tank digerakkan pemain lewat keyboard.
Mode kejar — begitu tank masuk radius pandang drone, drone menghitung ulang jalur tiap giliran memakai algoritma yang dipilih di panel kanan (UCS atau A* dengan salah satu dari 3 heuristik).
Mode investigasi — kalau kontak terputus (tank keluar radius pandang), drone mengingat posisi terakhir tank terlihat, mendatangi titik itu dulu, baru melanjutkan menyisir dari titik terdekat (bukan balik ke posisi awal).
Kontrol
Aksi	Tombol
Gerakkan tank	Panah atas/bawah/kiri/kanan atau W A S D
Ganti algoritma drone	Dropdown "Algoritma pencarian" (UCS / A*)
Ganti heuristik A*	Dropdown "Heuristik" (Manhattan / Euclidean / Chebyshev)
Ubah radius pandang drone	Dropdown "Radius pandang drone"
Drone bergerak otomatis	Centang "Drone bergerak otomatis tiap 450ms"
Acak medan baru	Tombol Acak Medan (peta + posisi awal tank & drone diacak ulang)
Kembali ke posisi awal	Tombol Reset Posisi (medan yang sama, posisi direset)
Bandingkan 4 kombinasi algoritma sekaligus	Tombol Jalankan Eksperimen
Jenis medan & biaya tempuh:
Medan	Tank	Drone (biaya)
Tanah	bisa lewat 1	
Pohon	terhalang	4
Bebatuan	terhalang	2
Sungai	terhalang kecuali di jembatan	2 (drone terbang melintasinya)
Jembatan	bisa lewat	1
Artileri anti-udara	terhalang	terhalang (satu-satunya yang menghalangi drone juga)
Kamp tentara	terhalang	6
Bangunan runtuh	terhalang	2

Catatan: karena drone digambarkan "terbang", ia bisa melintasi rintangan darat (pohon, batu, sungai, kamp, reruntuhan) dengan biaya lebih mahal — kecuali artileri anti-udara, yang menghalangi tank dan drone.

Pembuatan medan (peta acak)

Urutan pembuatan tiap kali "Acak Medan" ditekan:

Sungai berkelok dibuat lewat random walk dari baris paling atas ke paling bawah (bisa turun lurus, diagonal, atau menyamping — lima arah dengan peluang sama rata).
Sejumlah baris sungai dipilih jadi titik jembatan; seluruh lebar sungai di baris itu dijadikan jembatan sekaligus (bukan cuma satu sel), supaya jembatan selalu menyambung darat ke darat tanpa sisa air di tengah.
Zona aman digambar di sekeliling tiap jembatan supaya tidak ada rintangan lain menempel dan menghalangi jalan masuknya.
Pohon (~8%), bebatuan (~6%), 5 artileri, 2 blok kamp (2x2), dan 2 blok bangunan runtuh (2x2) disebar secara acak di sisa tanah kosong.
Peta divalidasi lewat BFS: titik tank dan drone harus tetap terhubung. Kalau tidak, seluruh medan dibuat ulang (maksimum 25 percobaan).
Panel statistik

Setiap kali drone menjalankan pencarian jalur nyata (bukan pola menyisir), panel kanan menampilkan: algoritma yang dipakai, apakah jalur ditemukan, total biaya jalur, jumlah node yang diekspansi, dan waktu komputasi (ms).

Tombol Jalankan Eksperimen membandingkan 4 kombinasi sekaligus (UCS, A*+Manhattan, A*+Euclidean, A*+Chebyshev) dari posisi drone ke tank saat ini, lalu menandai kombinasi paling efisien (node yang diekspansi paling sedikit).

Catatan:
Sebenarnya kami tidak mengerjakan di repo ini, tapi karena satu dan lain hal (seperti commit message yang sangat tidak rapi) kami memutuskan untuk membuat repo baru
