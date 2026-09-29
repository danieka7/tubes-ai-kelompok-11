# Tank vs Drone — Simulasi Pathfinding & Adversarial Search

Simulasi berbasis **Python + Pygame** yang mendemonstrasikan dua kelas algoritma AI dalam satu permainan:

1. **Pathfinding** (UCS dan A\*) — drone mencari dan mengejar tank di medan perang berbentuk grid.
2. **Adversarial search** (Minimax, Alpha-Beta Pruning, Expectimax) — saat drone cukup dekat, permainan berpindah ke mode *battle* bergiliran antara tank (pemain) dan drone (NPC).

Proyek ini dibuat untuk keperluan pembelajaran algoritma pencarian dan dapat dipakai sebagai bahan eksperimen perbandingan antar algoritma.

---

## Daftar Isi

- [Tank vs Drone — Simulasi Pathfinding \& Adversarial Search](#tank-vs-drone--simulasi-pathfinding--adversarial-search)
  - [Daftar Isi](#daftar-isi)
  - [Fitur](#fitur)
    - [Mode Eksplorasi (Pathfinding)](#mode-eksplorasi-pathfinding)
    - [Mode Battle (Adversarial Search)](#mode-battle-adversarial-search)
    - [Antarmuka](#antarmuka)
  - [Persyaratan](#persyaratan)
  - [Instalasi \& Menjalankan](#instalasi--menjalankan)
  - [Kontrol](#kontrol)
    - [Mode Eksplorasi](#mode-eksplorasi)
    - [Mode Battle (semua lewat klik tombol)](#mode-battle-semua-lewat-klik-tombol)
    - [Umum](#umum)
  - [Cara Kerja](#cara-kerja)
    - [Mode Eksplorasi](#mode-eksplorasi-1)
    - [Mode Battle](#mode-battle)
  - [Jenis Medan](#jenis-medan)
  - [Eksperimen Bawaan](#eksperimen-bawaan)
  - [Struktur Kode](#struktur-kode)
  - [Konfigurasi](#konfigurasi)
  - [Catatan](#catatan)
  - [Lisensi](#lisensi)
  - [Daftar Isi](#daftar-isi-1)
  - [Fitur](#fitur-1)
    - [Mode Eksplorasi (Pathfinding)](#mode-eksplorasi-pathfinding-1)
    - [Mode Battle (Adversarial Search)](#mode-battle-adversarial-search-1)
    - [Antarmuka](#antarmuka-1)
  - [Persyaratan](#persyaratan-1)
  - [Instalasi \& Menjalankan](#instalasi--menjalankan-1)
  - [Kontrol](#kontrol-1)
    - [Mode Eksplorasi](#mode-eksplorasi-2)
    - [Mode Battle (semua lewat klik tombol)](#mode-battle-semua-lewat-klik-tombol-1)
    - [Umum](#umum-1)
  - [Cara Kerja](#cara-kerja-1)
    - [Mode Eksplorasi](#mode-eksplorasi-3)
    - [Mode Battle](#mode-battle-1)
  - [Jenis Medan](#jenis-medan-1)
  - [Eksperimen Bawaan](#eksperimen-bawaan-1)
  - [Struktur Kode](#struktur-kode-1)
  - [Konfigurasi](#konfigurasi-1)
  - [Catatan](#catatan-1)
  - [Lisensi](#lisensi-1)

---

## Fitur

### Mode Eksplorasi (Pathfinding)
- Medan acak: sungai, jembatan, pohon, batu, artileri anti-udara, kamp, dan bangunan runtuh.
- Algoritma **UCS** dan **A\*** dengan tiga heuristik: Manhattan, Euclidean, Chebyshev.
- Drone punya beberapa status perilaku: memindai, bergerak, menyisir, mengejar, dan menyelidiki posisi terakhir tank.
- Pola sisiran **boustrophedon** saat posisi tank belum diketahui.
- Animasi gelombang ekspansi UCS/A\* pada pemindaian awal.
- Panel statistik (biaya jalur, node diekspansi, waktu komputasi) dan eksperimen perbandingan heuristik.

### Mode Battle (Adversarial Search)
- Pertempuran bergiliran: **Serang**, **Bertahan**, **Reparasi**.
- NPC memakai **Minimax**, **Alpha-Beta Pruning**, atau **Expectimax** (dipilih lewat tombol).
- Empat fungsi evaluasi dengan "kepribadian" berbeda: Seimbang, Agresif, Defensif, Hemat Reparasi.
- Tiga urutan aksi (*move ordering*) untuk mengamati efek terhadap efisiensi pemangkasan.
- Kedalaman pencarian 1–6.
- **Visualisasi pohon pencarian** interaktif: arahkan mouse ke sebuah node untuk melihat detail (aksi, pemilik node, nilai, dan snapshot state).
- **Tabel eksperimen** yang membandingkan algoritma, fungsi evaluasi, urutan aksi, dan kedalaman sekaligus.
- Setelah pertempuran selesai, hasil tetap ditampilkan dan pemain memilih sendiri kapan kembali ke mode eksplorasi lewat tombol konfirmasi.

### Antarmuka
- Jendela dapat diperbesar (*resizable*); tata letak kedua mode menyesuaikan ukuran jendela.
- Layar penuh dengan tombol **F11**.

---

## Persyaratan

- Python **3.8** atau lebih baru
- [Pygame](https://www.pygame.org/) **2.0** atau lebih baru

---

## Instalasi & Menjalankan

```bash
# 1. Clone repositori
git clone https://github.com/<username>/<nama-repositori>.git
cd <nama-repositori>

# 2. (Opsional) buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Pasang dependensi
pip install pygame

# 4. Jalankan
python <nama-file>.py
```

> Ganti `<username>`, `<nama-repositori>`, dan `<nama-file>` sesuai repositori dan nama file kamu.

---

## Kontrol

### Mode Eksplorasi

| Input                                        | Fungsi                                      |
| -------------------------------------------- | ------------------------------------------- |
| Panah / `W` `A` `S` `D`                      | Menggerakkan tank                           |
| Tombol **UCS** / **A\***                     | Memilih algoritma pencarian drone           |
| Tombol **Manhattan / Euclidean / Chebyshev** | Memilih heuristik (hanya aktif untuk A\*)   |
| Tombol **2 / 3 / 4 / 6 / 8**                 | Mengatur radius pandang drone               |
| Tombol **Drone bergerak otomatis**           | Drone bergerak sendiri tiap 450 ms          |
| **Acak Medan**                               | Membuat medan dan posisi awal baru          |
| **Reset Posisi**                             | Mengembalikan tank dan drone ke posisi awal |
| **Jalankan Eksperimen**                      | Membandingkan UCS dan A\* (tiga heuristik)  |

### Mode Battle (semua lewat klik tombol)

| Tombol                                             | Fungsi                                   |
| -------------------------------------------------- | ---------------------------------------- |
| **[Serang] / [Bertahan] / [Reparasi]**             | Aksi pemain                              |
| **Minimax / Alpha-Beta / Expectimax**              | Algoritma NPC                            |
| **Seimbang / Agresif / Defensif / Hemat Reparasi** | Fungsi evaluasi NPC                      |
| **Serang dulu / Bertahan dulu / Reparasi dulu**    | Urutan aksi (*move ordering*)            |
| **1–6** (baris pertama)                            | Kedalaman pencarian (`MAX_DEPTH`)        |
| **1–6** (baris kedua)                              | Kedalaman pohon yang digambar            |
| **Pohon Pencarian / Tabel Eksperimen**             | Mengganti tampilan panel kanan           |
| **Jalankan Eksperimen**                            | Menjalankan semua perbandingan sekaligus |
| **Kembali ke Mode Eksplorasi**                     | Muncul setelah pertempuran selesai       |

### Umum

| Input | Fungsi      |
| ----- | ----------- |
| `F11` | Layar penuh |
| `Esc` | Keluar      |

---

## Cara Kerja

### Mode Eksplorasi

Drone dan tank adalah dua agen dengan aturan gerak berbeda:

- **Tank** hanya bisa melewati tanah dan jembatan.
- **Drone** bisa terbang melewati hampir semua medan dengan biaya berbeda, kecuali artileri anti-udara.

Alur perilaku drone:

```
MEMINDAI  →  BERGERAK  →  MENYISIR  ⇄  MENGEJAR
                              ↓            ↑
                         MENYELIDIKI ──────┘
```

1. **Memindai** — UCS/A\* dijalankan ke posisi awal tank, dan gelombang ekspansinya dianimasikan.
2. **Bergerak** — drone menuju posisi awal tank (mengabaikan pergerakan tank sementara).
3. **Menyisir** — bila tank tidak terlihat, drone mengikuti pola boustrophedon.
4. **Mengejar** — bila tank masuk radius pandang dan *line of sight* tidak terhalang.
5. **Menyelidiki** — bila kontak terputus, drone menuju posisi terakhir tank diketahui.

Saat jarak Manhattan drone–tank mencapai ambang (`BATTLE_TRIGGER_DISTANCE`), permainan masuk ke mode battle.

### Mode Battle

State battle: HP, jumlah kit reparasi, status bertahan, dan giliran.

| Aksi     | Efek                                                         |
| -------- | ------------------------------------------------------------ |
| Serang   | Memberi 18 damage (dikurangi 50% bila lawan sedang bertahan) |
| Bertahan | Damage yang masuk pada giliran berikutnya berkurang 50%      |
| Reparasi | Memulihkan 30 HP, memakai satu kit (awal: 3 kit)             |

Fungsi utilitas terminal (dari sudut pandang NPC):

```
U(s) = +1000 - turnIndex   jika HP pemain <= 0   (NPC menang)
U(s) = -1000 + turnIndex   jika HP NPC <= 0      (pemain menang)
```

Pada batas kedalaman (*cutoff*), node dinilai dengan fungsi evaluasi yang dipilih. Ini adalah asumsi desain: NPC lebih menyukai kemenangan cepat dan kekalahan lambat.

Perbedaan ketiga algoritma:

- **Minimax** — asumsi lawan bermain optimal, tanpa pemangkasan.
- **Alpha-Beta** — hasil sama dengan Minimax, tetapi memangkas cabang yang tidak berpengaruh.
- **Expectimax** — giliran pemain dianggap acak (semua aksi legal sama mungkin), NPC memaksimalkan nilai harapan.

---

## Jenis Medan

| Medan               | Tank      | Drone (biaya) |
| ------------------- | --------- | ------------- |
| Tanah               | Lewat     | 1             |
| Jembatan            | Lewat     | 1             |
| Bebatuan            | Terhalang | 2             |
| Sungai              | Terhalang | 2             |
| Bangunan runtuh     | Terhalang | 2             |
| Pohon               | Terhalang | 4             |
| Kamp tentara        | Terhalang | 6             |
| Artileri anti-udara | Terhalang | Terhalang     |

Pohon, batu, kamp, dan bangunan runtuh juga menghalangi *line of sight*.

---

## Eksperimen Bawaan

**Mode Eksplorasi** membandingkan UCS, A\* + Manhattan, A\* + Euclidean, dan A\* + Chebyshev pada posisi saat ini (jumlah node, biaya jalur, waktu).

**Mode Battle** menjalankan empat kelompok perbandingan dari state saat ini:

1. Algoritma pencarian (Minimax vs Alpha-Beta vs Expectimax)
2. Fungsi evaluasi (perbedaan aksi terpilih = "kepribadian" NPC)
3. Urutan aksi (pengaruh *move ordering* terhadap jumlah node yang dipangkas)
4. Kedalaman 1–6 (pertumbuhan node dan waktu)

---

## Struktur Kode

Saat ini seluruh program berada dalam **satu file** dan dibagi menjadi beberapa bagian:

| Bagian | Isi                                                                                                      |
| ------ | -------------------------------------------------------------------------------------------------------- |
| 1      | Konstanta dunia dan jenis medan                                                                          |
| 2      | Pembangkit medan acak dan posisi awal                                                                    |
| 3      | UCS / A\* (`search`) dan heuristik                                                                       |
| 4      | Warna, konstanta tampilan, dan `apply_layout`                                                            |
| 5      | Widget UI (`Button`, `wrap_text`)                                                                        |
| 5B     | Mode battle: `BattleState`, `TreeNode`, `MinimaxAgent`, fungsi evaluasi, `BattleController`, `BattleHUD` |
| 6      | `Game` (kelas utama)                                                                                     |
| 7      | `main()`                                                                                                 |

Kelas utama:

- `SearchResult` — hasil pencarian jalur
- `Button` — widget tombol
- `BattleState` — state battle (immutable-style: `apply_action` mengembalikan state baru)
- `TreeNode` — node pohon pencarian untuk visualisasi
- `MinimaxAgent` — Minimax / Alpha-Beta / Expectimax
- `BattleController` — alur giliran battle dan tombol-tombolnya
- `BattleHUD` — penggambaran layar battle
- `Game` — mesin mode eksplorasi dan pengalih mode

---

## Konfigurasi

Beberapa konstanta yang mudah disesuaikan:

| Konstanta                  | Fungsi                        | Bawaan  |
| -------------------------- | ----------------------------- | ------- |
| `ROWS`, `COLS`             | Ukuran grid                   | 16 × 22 |
| `BASE_CELL`                | Ukuran petak minimum (piksel) | 30      |
| `BATTLE_TRIGGER_DISTANCE`  | Jarak Manhattan pemicu battle | 1       |
| `BATTLE_DEFAULT_DEPTH`     | Kedalaman pencarian awal      | 4       |
| `ATTACK_DAMAGE`            | Damage serangan               | 18      |
| `REPAIR_HEAL_AMOUNT`       | HP pulih per reparasi         | 30      |
| `BATTLE_START_REPAIR_KITS` | Kit reparasi awal             | 3       |
| `AUTO_CHASE_TICK_MS`       | Interval gerak otomatis drone | 450     |

---

## Catatan

- Bagian visual sengaja disederhanakan dengan primitif Pygame (rect, circle, line).
- Ukuran font tidak ikut membesar saat jendela diperbesar; hanya ruang tata letaknya.
- Pohon pencarian yang digambar diturunkan kedalamannya otomatis bila terlalu lebar agar tetap terbaca.

1. **Pathfinding** (UCS dan A\*) — drone mencari dan mengejar tank di medan perang berbentuk grid.
2. **Adversarial search** (Minimax, Alpha-Beta Pruning, Expectimax) — saat drone cukup dekat, permainan berpindah ke mode *battle* bergiliran antara tank (pemain) dan drone (NPC).

Proyek ini dibuat untuk memenuhi tugas besar dalam mata kuliah kecerdasa buatan.

---

## Daftar Isi

- [Fitur](#fitur)
- [Persyaratan](#persyaratan)
- [Instalasi & Menjalankan](#instalasi--menjalankan)
- [Kontrol](#kontrol)
- [Cara Kerja](#cara-kerja)
  - [Mode Eksplorasi](#mode-eksplorasi)
  - [Mode Battle](#mode-battle)
- [Jenis Medan](#jenis-medan)
- [Eksperimen Bawaan](#eksperimen-bawaan)
- [Struktur Kode](#struktur-kode)
- [Konfigurasi](#konfigurasi)
- [Catatan](#catatan)

---

## Fitur

### Mode Eksplorasi (Pathfinding)
- Medan acak: sungai, jembatan, pohon, batu, artileri anti-udara, kamp, dan bangunan runtuh.
- Algoritma **UCS** dan **A\*** dengan tiga heuristik: Manhattan, Euclidean, Chebyshev.
- Drone punya beberapa status perilaku: memindai, bergerak, menyisir, mengejar, dan menyelidiki posisi terakhir tank.
- Pola sisiran **boustrophedon** saat posisi tank belum diketahui.
- Animasi gelombang ekspansi UCS/A\* pada pemindaian awal.
- Panel statistik (biaya jalur, node diekspansi, waktu komputasi) dan eksperimen perbandingan heuristik.

### Mode Battle (Adversarial Search)
- Pertempuran bergiliran: **Serang**, **Bertahan**, **Reparasi**.
- NPC memakai **Minimax**, **Alpha-Beta Pruning**, atau **Expectimax** (dipilih lewat tombol).
- Empat fungsi evaluasi dengan "kepribadian" berbeda: Seimbang, Agresif, Defensif, Hemat Reparasi.
- Tiga urutan aksi (*move ordering*) untuk mengamati efek terhadap efisiensi pemangkasan.
- Kedalaman pencarian 1–6.
- **Visualisasi pohon pencarian** interaktif: arahkan mouse ke sebuah node untuk melihat detail (aksi, pemilik node, nilai, dan snapshot state).
- **Tabel eksperimen** yang membandingkan algoritma, fungsi evaluasi, urutan aksi, dan kedalaman sekaligus.
- Setelah pertempuran selesai, hasil tetap ditampilkan dan pemain memilih sendiri kapan kembali ke mode eksplorasi lewat tombol konfirmasi.

### Antarmuka
- Jendela dapat diperbesar (*resizable*); tata letak kedua mode menyesuaikan ukuran jendela.
- Layar penuh dengan tombol **F11**.

---

## Persyaratan

- Python **3.8** atau lebih baru
- [Pygame](https://www.pygame.org/) **2.0** atau lebih baru

---

## Instalasi & Menjalankan

```bash
# 1. Clone repositori
git clone https://github.com/<username>/<nama-repositori>.git
cd <nama-repositori>

# 2. (Opsional) buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Pasang dependensi
pip install pygame

# 4. Jalankan
python <nama-file>.py
```

> Ganti `<username>`, `<nama-repositori>`, dan `<nama-file>` sesuai repositori dan nama file kamu.

---

## Kontrol

### Mode Eksplorasi

| Input | Fungsi |
|-------|--------|
| Panah / `W` `A` `S` `D` | Menggerakkan tank |
| Tombol **UCS** / **A\*** | Memilih algoritma pencarian drone |
| Tombol **Manhattan / Euclidean / Chebyshev** | Memilih heuristik (hanya aktif untuk A\*) |
| Tombol **2 / 3 / 4 / 6 / 8** | Mengatur radius pandang drone |
| Tombol **Drone bergerak otomatis** | Drone bergerak sendiri tiap 450 ms |
| **Acak Medan** | Membuat medan dan posisi awal baru |
| **Reset Posisi** | Mengembalikan tank dan drone ke posisi awal |
| **Jalankan Eksperimen** | Membandingkan UCS dan A\* (tiga heuristik) |

### Mode Battle (semua lewat klik tombol)

| Tombol | Fungsi |
|--------|--------|
| **[Serang] / [Bertahan] / [Reparasi]** | Aksi pemain |
| **Minimax / Alpha-Beta / Expectimax** | Algoritma NPC |
| **Seimbang / Agresif / Defensif / Hemat Reparasi** | Fungsi evaluasi NPC |
| **Serang dulu / Bertahan dulu / Reparasi dulu** | Urutan aksi (*move ordering*) |
| **1–6** (baris pertama) | Kedalaman pencarian (`MAX_DEPTH`) |
| **1–6** (baris kedua) | Kedalaman pohon yang digambar |
| **Pohon Pencarian / Tabel Eksperimen** | Mengganti tampilan panel kanan |
| **Jalankan Eksperimen** | Menjalankan semua perbandingan sekaligus |
| **Kembali ke Mode Eksplorasi** | Muncul setelah pertempuran selesai |

### Umum

| Input | Fungsi |
|-------|--------|
| `F11` | Layar penuh |
| `Esc` | Keluar |

---

## Cara Kerja

### Mode Eksplorasi

Drone dan tank adalah dua agen dengan aturan gerak berbeda:

- **Tank** hanya bisa melewati tanah dan jembatan.
- **Drone** bisa terbang melewati hampir semua medan dengan biaya berbeda, kecuali artileri anti-udara.

Alur perilaku drone:

```
MEMINDAI  →  BERGERAK  →  MENYISIR  ⇄  MENGEJAR
                              ↓            ↑
                         MENYELIDIKI ──────┘
```

1. **Memindai** — UCS/A\* dijalankan ke posisi awal tank, dan gelombang ekspansinya dianimasikan.
2. **Bergerak** — drone menuju posisi awal tank (mengabaikan pergerakan tank sementara).
3. **Menyisir** — bila tank tidak terlihat, drone mengikuti pola boustrophedon.
4. **Mengejar** — bila tank masuk radius pandang dan *line of sight* tidak terhalang.
5. **Menyelidiki** — bila kontak terputus, drone menuju posisi terakhir tank diketahui.

Saat jarak Manhattan drone–tank mencapai ambang (`BATTLE_TRIGGER_DISTANCE`), permainan masuk ke mode battle.

### Mode Battle

State battle: HP, jumlah kit reparasi, status bertahan, dan giliran.

| Aksi | Efek |
|------|------|
| Serang | Memberi 18 damage (dikurangi 50% bila lawan sedang bertahan) |
| Bertahan | Damage yang masuk pada giliran berikutnya berkurang 50% |
| Reparasi | Memulihkan 30 HP, memakai satu kit (awal: 3 kit) |

Fungsi utilitas terminal (dari sudut pandang NPC):

```
U(s) = +1000 - turnIndex   jika HP pemain <= 0   (NPC menang)
U(s) = -1000 + turnIndex   jika HP NPC <= 0      (pemain menang)
```

Pada batas kedalaman (*cutoff*), node dinilai dengan fungsi evaluasi yang dipilih. Ini adalah asumsi desain: NPC lebih menyukai kemenangan cepat dan kekalahan lambat.

Perbedaan ketiga algoritma:

- **Minimax** — asumsi lawan bermain optimal, tanpa pemangkasan.
- **Alpha-Beta** — hasil sama dengan Minimax, tetapi memangkas cabang yang tidak berpengaruh.
- **Expectimax** — giliran pemain dianggap acak (semua aksi legal sama mungkin), NPC memaksimalkan nilai harapan.

---

## Jenis Medan

| Medan | Tank | Drone (biaya) |
|-------|------|---------------|
| Tanah | Lewat | 1 |
| Jembatan | Lewat | 1 |
| Bebatuan | Terhalang | 2 |
| Sungai | Terhalang | 2 |
| Bangunan runtuh | Terhalang | 2 |
| Pohon | Terhalang | 4 |
| Kamp tentara | Terhalang | 6 |
| Artileri anti-udara | Terhalang | Terhalang |

Pohon, batu, kamp, dan bangunan runtuh juga menghalangi *line of sight*.

---

## Eksperimen Bawaan

**Mode Eksplorasi** membandingkan UCS, A\* + Manhattan, A\* + Euclidean, dan A\* + Chebyshev pada posisi saat ini (jumlah node, biaya jalur, waktu).

**Mode Battle** menjalankan empat kelompok perbandingan dari state saat ini:

1. Algoritma pencarian (Minimax vs Alpha-Beta vs Expectimax)
2. Fungsi evaluasi (perbedaan aksi terpilih = "kepribadian" NPC)
3. Urutan aksi (pengaruh *move ordering* terhadap jumlah node yang dipangkas)
4. Kedalaman 1–6 (pertumbuhan node dan waktu)

---

## Struktur Kode

Saat ini seluruh program berada dalam **satu file** dan dibagi menjadi beberapa bagian:

| Bagian | Isi |
|--------|-----|
| 1 | Konstanta dunia dan jenis medan |
| 2 | Pembangkit medan acak dan posisi awal |
| 3 | UCS / A\* (`search`) dan heuristik |
| 4 | Warna, konstanta tampilan, dan `apply_layout` |
| 5 | Widget UI (`Button`, `wrap_text`) |
| 5B | Mode battle: `BattleState`, `TreeNode`, `MinimaxAgent`, fungsi evaluasi, `BattleController`, `BattleHUD` |
| 6 | `Game` (kelas utama) |
| 7 | `main()` |

Kelas utama:

- `SearchResult` — hasil pencarian jalur
- `Button` — widget tombol
- `BattleState` — state battle (immutable-style: `apply_action` mengembalikan state baru)
- `TreeNode` — node pohon pencarian untuk visualisasi
- `MinimaxAgent` — Minimax / Alpha-Beta / Expectimax
- `BattleController` — alur giliran battle dan tombol-tombolnya
- `BattleHUD` — penggambaran layar battle
- `Game` — mesin mode eksplorasi dan pengalih mode

---

## Konfigurasi

Beberapa konstanta yang mudah disesuaikan:

| Konstanta | Fungsi | Bawaan |
|-----------|--------|--------|
| `ROWS`, `COLS` | Ukuran grid | 16 × 22 |
| `BASE_CELL` | Ukuran petak minimum (piksel) | 30 |
| `BATTLE_TRIGGER_DISTANCE` | Jarak Manhattan pemicu battle | 1 |
| `BATTLE_DEFAULT_DEPTH` | Kedalaman pencarian awal | 4 |
| `ATTACK_DAMAGE` | Damage serangan | 18 |
| `REPAIR_HEAL_AMOUNT` | HP pulih per reparasi | 30 |
| `BATTLE_START_REPAIR_KITS` | Kit reparasi awal | 3 |
| `AUTO_CHASE_TICK_MS` | Interval gerak otomatis drone | 450 |

---

## Catatan

- Bagian visual sengaja disederhanakan dengan primitif Pygame (rect, circle, line).
- Ukuran font tidak ikut membesar saat jendela diperbesar; hanya ruang tata letaknya.
- Pohon pencarian yang digambar diturunkan kedalamannya otomatis bila terlalu lebar agar tetap terbaca.

## Lisensi

Tambahkan lisensi pilihanmu di sini (misalnya MIT) dan sertakan berkas `LICENSE` di repositori.
