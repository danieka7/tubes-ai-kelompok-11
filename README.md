# Tank vs Drone Simulasi Ukraina  — Local Search dan CSP

Proyek simulasi *Tank vs Drone* berbasis Python dan Pygame untuk mendemonstrasikan penerapan algoritma pencarian pada agen AI. Fokus Tahap 3 adalah **Constraint Satisfaction Problem (CSP)** untuk pembangkitan peta yang valid dan **local search** untuk optimasi keseimbangan pertempuran serta pemilihan aksi NPC.

Implementasi tambahan mencakup pencarian jalur drone dengan Uniform Cost Search (UCS) dan A*, serta pengambilan keputusan adversarial melalui Minimax, Alpha-Beta Pruning, dan Expectimax. Komponen tersebut melengkapi simulasi, sementara pembahasan utama README ini berada pada CSP dan local search.

## Fokus Implementasi Tahap 3

### 1. Pembangkitan Peta Menggunakan CSP

Peta dibangun pada grid berukuran **16 × 22 sel**. Proses pembangkitan membuat jalur sungai dan jembatan terlebih dahulu, kemudian menempatkan objek medan menggunakan CSP. Peta dasar diperiksa agar tank masih memiliki jalur dari posisi awal menuju tujuan.

#### Formulasi CSP

| Komponen | Implementasi |
| --- | --- |
| Variabel | Satu variabel untuk setiap objek camp, ruin, artileri, pohon, dan batu. |
| Domain | Posisi *anchor* berupa pasangan baris-kolom. Untuk objek berukuran 2 × 2, anchor menunjukkan sel kiri-atas footprint objek. |
| Constraint unary | Sel yang ditempati harus berada di atas medan dasar `DIRT`, tidak boleh menempati posisi awal/tujuan, dan harus berada di luar zona larangan di sekitar jembatan. |
| Constraint non-overlap | Footprint objek tidak boleh saling bertumpang tindih. |
| Constraint kardinalitas | Jumlah objek per jenis ditetapkan oleh konfigurasi pembangkitan. |
| Constraint global | Setelah penempatan objek, tank harus tetap dapat mencapai tujuan. |

Jumlah variabel diturunkan dari konfigurasi saat ini: 2 camp, 2 ruin, 5 artileri, sekitar 8% dari total sel untuk pohon, dan sekitar 6% dari total sel untuk batu. Dengan grid 16 × 22, konfigurasi tersebut menghasilkan **58 variabel objek**.

#### Strategi pencarian CSP

Solver `MapCspSolver` menggunakan kombinasi teknik berikut:

- **Backtracking** — mencoba penempatan objek dan kembali ke keputusan sebelumnya jika cabang tidak menghasilkan solusi.
- **Minimum Remaining Values (MRV)** — memilih variabel dengan domain tersisa paling kecil agar variabel yang paling terbatas diproses lebih dahulu.
- **Degree heuristic** — memecah seri MRV dengan memprioritaskan variabel yang diperkirakan memiliki konflik lebih besar dengan variabel yang belum ditetapkan.
- **Least Constraining Value (LCV)** — mencoba posisi yang diperkirakan menghilangkan paling sedikit pilihan pada domain variabel lain. Skornya diberi sedikit jitter acak agar hasil tidak selalu bias ke posisi yang sama.
- **Forward checking** — setelah sebuah objek ditempatkan, menghapus posisi yang bertabrakan dari domain variabel yang belum ditetapkan.
- **Reachability pruning** — menolak cabang segera ketika penempatan objek membuat jalur tank ke tujuan terputus.

Pembangkitan dibatasi oleh maksimum **25 percobaan peta** dan batas **20.000 node penempatan** untuk menjaga proses pencarian tetap terkendali. Jika peta valid tidak ditemukan dalam batas tersebut, program menghasilkan `MapGenerationError`.

#### Statistik CSP

Panel debug menampilkan metrik yang dapat digunakan untuk mengamati proses solver, antara lain waktu pembangkitan, jumlah percobaan peta, jumlah variabel dan ukuran domain awal, node yang dikunjungi, kedalaman maksimum, jumlah backtrack, nilai domain yang dipangkas, *domain wipeout*, pemeriksaan keterjangkauan, cabang yang dipangkas, serta jumlah objek yang berhasil ditempatkan.

Metrik ini membantu mengevaluasi bukan hanya apakah peta berhasil dibuat, tetapi juga seberapa banyak pencarian dan pemangkasan yang terjadi selama prosesnya.

### 2. Local Search untuk Optimasi Keseimbangan Pertempuran

Program menggunakan local search untuk mencari konfigurasi parameter pertempuran yang mendekati sasaran keseimbangan, bukan hanya mengandalkan penentuan nilai secara manual.

#### Ruang pencarian

State pencarian adalah objek `BalanceParams`, yang memuat sepuluh parameter, yaitu HP tank; jumlah drone awal; HP dan damage drone Bom, Misil, dan Kamikaze; ronde mulai diperbolehkannya bantuan; serta peluang kemunculan bantuan.

Satu *neighbor* dibuat dengan mengubah **satu parameter** sebesar satu atau dua langkah konfigurasi, lalu memastikan nilainya tetap berada di rentang yang diizinkan.

#### Fungsi objektif (*loss function*)

Setiap kandidat parameter dievaluasi melalui simulasi pertempuran menggunakan dua kebijakan pemain: pemain biasa (*casual*) dan pemain mahir (*skilled*). Nilai loss yang lebih kecil berarti hasil simulasi semakin dekat dengan sasaran keseimbangan.

| Metrik | Sasaran |
| --- | --- |
| Tingkat kemenangan pemain biasa | 35–55% |
| Tingkat kemenangan pemain mahir | 65–85% |
| Rata-rata durasi pertempuran | 10–22 ronde |
| Frekuensi pertempuran yang memunculkan bantuan | 25–60% |
| Porsi damage minimum tiap loadout | Minimal 20% dari total damage yang tercatat |
| Simulasi macet / timeout | Ditekan serendah mungkin |

Loss menggabungkan pelanggaran terhadap rentang target dengan bobot berbeda. Selisih tingkat kemenangan dan timeout mendapat bobot relatif besar; durasi, kemunculan bantuan, dan kontribusi loadout juga diperhitungkan. Nilai loss **0** berarti seluruh target yang dimodelkan terpenuhi.

#### Algoritma tuning

Fungsi `run_local_search()` menerapkan **stochastic first-improvement hill climbing dengan random restart**:

1. Restart pertama menggunakan `TUNED_PARAMS` saat ini, sedangkan restart berikutnya dimulai dari konfigurasi acak.
2. Algoritma membangkitkan neighbor dan menjalankan simulasi untuk menghitung loss kandidat.
3. Kandidat diterima hanya jika loss-nya lebih rendah daripada konfigurasi saat ini.
4. Pencarian berhenti ketika loss mencapai 0 atau setelah 25 neighbor berturut-turut tidak memperbaiki hasil.
5. Hasil terbaik dari seluruh restart dipilih dan dievaluasi kembali menggunakan seed validasi terpisah.

Evaluasi pencarian menjalankan 60 simulasi untuk masing-masing kebijakan pemain per kandidat. Untuk mengurangi pengaruh variasi acak terhadap perbandingan kandidat, seed evaluasi dibuat tetap. Setelah pencarian selesai, program menjalankan validasi dengan **400 simulasi per kebijakan pemain**.

### 3. Local Search untuk Pengambilan Keputusan NPC

Selain tuning parameter secara offline, local search juga tersedia sebagai algoritma agen NPC ketika pertempuran berlangsung. Pada mode ini, ruang pencarian bukan parameter keseimbangan, melainkan **rencana aksi untuk beberapa ronde ke depan**.

Sebuah rencana berisi urutan aksi `ATTACK`, `DEFEND`, atau `REPAIR`. Neighbor dibentuk dengan mengganti satu slot aksi. Rencana disimulasikan terhadap model tank sederhana: tank melakukan reparasi jika HP berada di bawah ambang yang ditentukan dan masih memiliki kit reparasi; jika tidak, tank menyerang. Skor akhir menggunakan utility jika duel telah berakhir atau fungsi evaluasi jika simulasi mencapai batas horizon.

Dua strategi tersedia:

| Algoritma | Cara kerja |
| --- | --- |
| **Hill Climbing** | Menerima neighbor hanya jika skornya lebih baik. Menggunakan tiga restart, dengan pembagian anggaran evaluasi untuk setiap restart. Karena hanya menerima perbaikan, algoritma dapat terjebak pada optimum lokal. |
| **Simulated Annealing** | Menggunakan aturan Metropolis: kandidat yang lebih baik diterima, sedangkan kandidat yang lebih buruk masih dapat diterima dengan probabilitas yang dipengaruhi suhu. Suhu dimulai dari 60 dan diturunkan secara geometrik dengan faktor 0,97. Mekanisme ini memberi kesempatan menjelajahi solusi yang tidak langsung lebih baik. |

Kedua mode menggunakan horizon default **3 ronde** dan anggaran **150 evaluasi rencana per keputusan**, agar perbandingan lebih sepadan. NPC hanya menjalankan aksi pertama dari rencana terbaik yang ditemukan. Pada giliran NPC berikutnya, rencana dihitung ulang berdasarkan state terbaru (*receding-horizon planning*).

Panel eksperimen local search menyediakan informasi seperti aksi dan skor terbaik, jumlah evaluasi, jumlah restart, jumlah perpindahan yang diterima, waktu pencarian, dan grafik konvergensi skor rencana yang sedang dipegang dibandingkan skor terbaik sejauh ini.

> **Catatan penting:** tuning parameter keseimbangan dan local search untuk agen NPC adalah dua proses yang berbeda. Opsi `--tune` menjalankan optimasi parameter secara offline di terminal; opsi tersebut tidak menjalankan jendela Pygame dan tidak otomatis menimpa konstanta `TUNED_PARAMS` di kode.

## Teknologi

- **Python 3.9+**
- **Pygame** untuk antarmuka, visualisasi grid, dan interaksi
- Modul standard library, termasuk `random`, `math`, `time`, `dataclasses`, dan `functools`

Tidak diperlukan framework frontend, database, atau berkas aset eksternal untuk menjalankan source yang disertakan.

## Instalasi dan Menjalankan Program

Pastikan Python tersedia di terminal. Disarankan menggunakan virtual environment agar dependensi proyek terisolasi.

### 1. Siapkan lingkungan

```bash
python -m venv .venv
```

Aktifkan environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Pasang dependensi:

```bash
python -m pip install pygame
```

### 2. Jalankan simulasi interaktif

Dengan nama source seperti berkas yang disertakan:

```bash
python "tahap 3(1).py"
```

Jika source diganti nama, sesuaikan nama berkas pada perintah tersebut. Jendela simulasi menyediakan mode eksplorasi dan battle, beserta panel untuk memilih dan mengamati algoritma AI.

### 3. Jalankan tuning local search dari terminal

Gunakan jumlah restart default:

```bash
python "tahap 3(1).py" --tune
```

Tentukan jumlah restart secara eksplisit, misalnya lima:

```bash
python "tahap 3(1).py" --tune 5
```

Proses ini menampilkan konfigurasi terbaik, nilai loss pencarian, statistik validasi, dan representasi `BalanceParams` yang dapat disalin ke konstanta `TUNED_PARAMS` jika hasilnya ingin digunakan pada simulasi berikutnya. Tuning dapat membutuhkan waktu karena setiap kandidat dinilai melalui banyak simulasi pertempuran.

## Struktur Kode

Source Tahap 3 saat ini dikemas dalam satu berkas Python. Bagian utama kode dapat dikenali melalui pembagian berikut:

| Bagian | Tanggung jawab |
| --- | --- |
| Konstanta dunia dan pembangkit medan | Ukuran grid, tipe medan, aturan pergerakan, sungai, dan jembatan. |
| CSP pembangkit peta | Variabel, domain, constraint, backtracking, heuristik, pruning, dan statistik CSP. |
| Pencarian jalur | UCS dan A* dengan biaya medan drone serta pilihan heuristik. |
| Sistem battle dan agen adversarial | Representasi state pertempuran, Minimax, Alpha-Beta Pruning, Expectimax, dan fungsi evaluasi. |
| Local search keseimbangan | Simulasi evaluasi parameter, fungsi loss, hill climbing, random restart, dan validasi. |
| Local search NPC | Representasi rencana aksi, Hill Climbing, Simulated Annealing, serta pencatatan konvergensi. |
| Antarmuka dan main loop | Visualisasi, panel statistik, kontrol interaksi, dan loop Pygame. |

## Batasan dan Interpretasi Hasil

- Hill climbing bersifat lokal; solusi terbaik yang ditemukan tidak dijamin sebagai optimum global.
- Simulated Annealing dapat menerima neighbor yang lebih buruk, tetapi hasilnya tetap dipengaruhi ruang pencarian, fungsi skor, jadwal pendinginan, dan anggaran evaluasi.
- Fungsi evaluasi dan target keseimbangan adalah asumsi desain untuk model simulasi, bukan ukuran universal tentang tingkat kesulitan permainan.
- Model tank yang dipakai saat mengevaluasi rencana NPC bersifat sederhana. Perilaku pemain nyata dapat berbeda dari model tersebut.
- Keberhasilan CSP menunjukkan bahwa penempatan memenuhi constraint yang diperiksa oleh implementasi; batas percobaan dan node berarti pembangkitan dapat gagal menemukan peta walaupun solusi lain mungkin ada.
- Statistik kinerja perlu dibandingkan pada konfigurasi, seed, dan anggaran evaluasi yang sepadan. Waktu eksekusi dapat berubah menurut lingkungan dan mesin.

## Tujuan Akademik

Tahap 3 memperlihatkan dua pendekatan optimasi yang berbeda. **CSP** memodelkan pembangkitan peta sebagai penetapan variabel dengan domain dan constraint eksplisit, sedangkan **local search** mencari konfigurasi atau rencana aksi yang memiliki nilai objektif lebih baik melalui perpindahan antar-solusi. Penggunaan statistik debug dan evaluasi simulasi membantu proses ini diamati dan dibandingkan secara kuantitatif.
