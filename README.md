# Tank vs Drone simulasi ukraina

Simulasi berbasis **Python + Pygame** yang mendemonstrasikan beberapa kelas algoritma AI dalam satu permainan:

1. **Pathfinding** (UCS dan A\*): drone mencari dan mengejar tank di medan perang berbentuk grid.
2. **Adversarial search** (Minimax, Alpha-Beta Pruning, Expectimax) dan **local search**: saat drone cukup dekat, permainan berpindah ke mode *battle* bergiliran.
3. **Logic & representasi pengetahuan** (materi 07): tiga KB FOL dengan resolution refutation; aturannya ada di `kb_rules.txt`.
4. **Automated planning** (materi 08): domain PDDL berparameter di atas peta, A\* maju, regresi mundur, dan planning graph dengan mutex.
5. **Decisions under uncertainty** (materi 09): MDP + value iteration, belief cuaca lewat aturan Bayes, dan kebijakan MEU.

Proyek ini dibuat untuk memenuhi tugas besar mata kuliah Kecerdasan Buatan.

---

## Daftar Isi

- [Fitur](#fitur)
- [Persyaratan](#persyaratan)
- [Instalasi & Menjalankan](#instalasi--menjalankan)
- [Kontrol](#kontrol)
- [Cara Kerja](#cara-kerja)
- [Logic: Basis Pengetahuan](#logic-basis-pengetahuan)
- [Automated Planning](#automated-planning)
- [Decisions under Uncertainty](#decisions-under-uncertainty)
- [Jenis Medan](#jenis-medan)
- [Eksperimen Bawaan](#eksperimen-bawaan)
- [Struktur Kode](#struktur-kode)
- [Konfigurasi](#konfigurasi)
- [Catatan](#catatan)

---

## Fitur

### Mode Eksplorasi (Pathfinding, KB, MDP)
- Medan acak: sungai, jembatan, pohon, batu, artileri anti-udara, kamp, dan bangunan runtuh.
- Algoritma **UCS** dan **A\*** dengan tiga heuristik: Manhattan, Euclidean, Chebyshev.
- Dua drone (A dan B) dengan status perilaku: memindai, bergerak, menyisir, mengejar, dan menyelidiki posisi terakhir tank.
- Pola sisiran **boustrophedon** saat posisi tank belum diketahui.
- Animasi gelombang ekspansi UCS/A\* pada pemindaian awal.
- **Penasihat tank (KB 1)**: klik sel di peta, lalu tanya komando "aman?" atau "terlihat?". Jawabannya YA / TIDAK / TIDAK PASTI, lengkap dengan bukti resolusi.
- **Interaksi NPC (KB 2)**: drone yang melihat tank membagikan kontak ke rekannya lewat inferensi `Investigate(e)`.
- **Mode MDP**: drone mengejar dengan kebijakan MEU di atas belief cuaca. Cuaca nyata tersembunyi, dan belief di-update lewat aturan Bayes dari gerakan yang meleset dan dari laporan cuaca.
- Panel statistik (biaya jalur, node diekspansi, waktu komputasi) dan eksperimen perbandingan heuristik.

### Mode Battle (Adversarial Search, Planning)
- Pertempuran bergiliran: **Serang**, **Bertahan**, **Reparasi**, melawan beberapa drone dengan loadout berbeda.
- NPC memakai **Minimax**, **Alpha-Beta Pruning**, **Expectimax**, **Hill Climbing**, atau **Simulated Annealing**.
- Empat fungsi evaluasi dengan "kepribadian" berbeda: Seimbang, Agresif, Defensif, Hemat Reparasi.
- Tiga urutan aksi (*move ordering*) dan kedalaman pencarian 1–6.
- **Visualisasi pohon pencarian** interaktif dan **tabel eksperimen**.
- **Komando musuh (KB 3 + planner)**: KB memutuskan kapan bantuan dipanggil dan peran apa yang dibutuhkan. Planner lalu menyusun rute unit dari pangkalan ke sektor tank, dan rencananya dieksekusi beberapa langkah per ronde. Semua prosesnya terlihat di tab **Komando (KB)**.

### Antarmuka
- Jendela dapat diperbesar (*resizable*); tata letak kedua mode menyesuaikan ukuran jendela.
- Layar penuh dengan tombol **F11**.
- Semua overlay inferensi menampilkan proses lengkap: KB beserta sumbernya, TELL fakta, kedua percobaan ASK (q lalu ~q), dan kesimpulan.

---

## Persyaratan

- Python **3.8** atau lebih baru
- [Pygame](https://www.pygame.org/) **2.0** atau lebih baru

---

## Instalasi & Menjalankan

```bash
pip install pygame

python tank_vs_drone_simulasi_ukraina.py                 # permainan
python tank_vs_drone_simulasi_ukraina.py --mdp-report    # analisis reward & profil MDP
python tank_vs_drone_simulasi_ukraina.py --tune 3        # local search keseimbangan
python -m unittest discover -s tests -v                  # tes unit KB, planner, MDP
```

Semua modul dan `kb_rules.txt` harus berada di folder yang sama.

---

## Kontrol

### Mode Eksplorasi

| Input                                        | Fungsi                                                          |
| -------------------------------------------- | --------------------------------------------------------------- |
| Panah / `W` `A` `S` `D`                      | Menggerakkan tank                                               |
| Tombol **UCS** / **A\*** / **MDP**           | Memilih cara drone mengejar                                     |
| Tombol **Manhattan / Euclidean / Chebyshev** | Memilih heuristik (hanya aktif untuk A\*)                       |
| Tombol **gamma 0.8 / 0.9 / 0.95**            | Discount MDP (baris heuristik berubah fungsi di mode MDP)       |
| **Nyata: Normal/Badai**                      | Mengganti cuaca sebenarnya (drone tidak diberi tahu)            |
| **Lapor cuaca**                              | Laporan cuaca dengan akurasi 80%, diikuti update Bayes          |
| **Tampil: Panah / Angka V / Mati**           | Tampilan kebijakan MEU atau V(s) harapan di peta                |
| **< Iterasi >** / **Akhir**                  | Menelusuri iterasi value iteration                              |
| Klik sel (mode MDP)                          | Overlay Bayes, EU tiap aksi, dan Bellman per cuaca              |
| **Tanya Komando** + **Aman? / Terlihat?**    | Klik sel untuk bertanya ke KB penasihat                         |
| **NPC (KB)**                                 | Overlay inferensi berbagi kontak antardrone                     |
| Tombol **2 / 3 / 4 / 6 / 8**                 | Mengatur radius pandang drone                                   |
| Tombol **Drone bergerak otomatis**           | Drone bergerak sendiri tiap 450 ms                              |
| **Acak Medan** / **Reset Posisi**            | Medan baru / posisi awal                                        |
| **Jalankan Eksperimen**                      | Membandingkan UCS dan A\* (tiga heuristik)                      |

### Mode Battle (semua lewat klik tombol)

| Tombol                                             | Fungsi                                   |
| -------------------------------------------------- | ---------------------------------------- |
| **[Serang] / [Bertahan] / [Reparasi]**             | Aksi pemain                              |
| **Minimax / Alpha-Beta / Expectimax / HC / SA**    | Algoritma NPC                            |
| **Seimbang / Agresif / Defensif / Hemat Reparasi** | Fungsi evaluasi NPC                      |
| **Serang dulu / Bertahan dulu / Reparasi dulu**    | Urutan aksi (*move ordering*)            |
| **1–6** (baris pertama)                            | Kedalaman pencarian (`MAX_DEPTH`)        |
| **1–6** (baris kedua)                              | Kedalaman pohon yang digambar            |
| **Pohon Pencarian / Tabel Eksperimen / Komando**   | Mengganti tampilan panel kanan           |
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

## Logic: Basis Pengetahuan

Basis pengetahuan **terpisah dari kode**. Semua aturan ditulis sebagai teks di [`kb_rules.txt`](kb_rules.txt), sedangkan mesin inferensinya ada di [`kb.py`](kb.py). Mesin itu mencakup parser, konversi CNF 4 langkah, unifikasi, dan resolution refutation dengan strategi set-of-support. File aturan dibaca saat program mulai, dan tiap formula langsung dicek sintaksnya. Bila file hilang atau satu bagian salah sintaks, bagian itu memakai aturan bawaan di `kb.py` (`DEFAULT_RULES`), dan alasannya ditampilkan di overlay sebagai "sumber: bawaan kode (...)".

| KB        | Pertanyaan (ASK)                      | Fakta persepsi (TELL)                                               |
| --------- | ------------------------------------- | ------------------------------------------------------------------- |
| `ADVISOR` | `Safe(sel)`, `Visible(sel)`           | `Passable`, `InRange`, `LineOfSight`, `Close` per drone, roster drone |
| `NPC`     | `Investigate(drone)`                  | `Spotted`, `Ally`, `Chasing`                                        |
| `COMMAND` | `CallReinforcement`, `Need<Peran>`    | 10 proposisi hasil `CommandCenter.sense()`                          |

Keputusan desain:

- **Bikonditional menggantikan aksioma penutup manual.** Versi lama memakai `R3` ditambah `R3a`–`R3e`, `R1a`, dan `R1b` untuk meniru "jika dan hanya jika". Sekarang cukup satu kalimat per konsep:
  `CallReinforcement <=> Outnumbered & ~Busy & GraceOver & SlotFree & QuotaLeft`.
  Konversi CNF menurunkan sendiri klausa-klausa penutupnya, sehingga KB bisa membuktikan `CallReinforcement` maupun `~CallReinforcement`. Hasilnya diuji pada seluruh 2^10 kombinasi persepsi dan sama dengan KB lama.
- **Tanpa nama drone di aturan.** "Sel tidak terlihat bila tidak ada drone yang melihatnya" adalah kuantor universal atas drone, yang tidak bisa ditulis sebagai premis klausa Horn. Karena itu roster drone dijalani berurutan:
  `FirstDrone(d) & Hidden(c,d) => HiddenUpTo(c,d)`, `HiddenUpTo(c,d) & NextDrone(d,e) & Hidden(c,e) => HiddenUpTo(c,e)`, `HiddenUpTo(c,d) & LastDrone(d) => NotVisible(c)`.
  Menambah drone ke-3 cukup dengan menambah fakta roster; aturannya tidak berubah (lihat tes `test_advisor_rules_scale_to_any_number_of_drones`). Drone di luar radar tidak punya fakta `Hidden`, sehingga rantainya putus dan jawabannya **TIDAK PASTI**.
- **Overlay selalu menampilkan proses lengkap** untuk ketiga KB: (1) aturan KB, jumlah klausa CNF, dan sumbernya; (2) fakta TELL; (3) ASK dalam dua percobaan, yaitu `q` lalu `~q`, masing-masing dengan negasi goal, langkah resolusi atau cuplikan pencarian yang gagal beserta alasannya (jenuh atau kena batas); (4) kesimpulan.

---

## Automated Planning

Bagian generiknya ada di [`planner.py`](planner.py), dan domain permainannya di [`command.py`](command.py).

**Domain logistik drone (PDDL berparameter).** Peta 22×16 dibagi menjadi 4×4 sektor. Sektor yang berisi kamp menjadi pangkalan, sektor yang berisi artileri anti-udara tertutup, dan sektor tank menjadi medan tempur. Ada tujuh skema aksi:

| Skema                          | Prekondisi (`*` = statik)                                                      | Efek                                  |
| ------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------- |
| `arm(?u ?b)`                   | `idle(?u)`, `at(?u,?b)`, `*base(?b)`                                            | `+armed(?u)` `-idle(?u)`              |
| `fly(?u ?a ?b)`                | `armed(?u)`, `at(?u,?a)`, `*adj(?a,?b)`, `*open(?b)`                            | `+at(?u,?b)` `-at(?u,?a)`             |
| `jam(?s)`                      | `ew-ready`, `*aa(?s)`                                                           | `+suppressed(?s)` `-ew-ready` (biaya 2) |
| `fly-jammed(?u ?a ?b)`         | `armed`, `at(?u,?a)`, `*adj`, `*aa(?b)`, `suppressed(?b)`                       | pindah sektor                         |
| `designate(?s)`                | `spotter-alive`, `*battle(?s)`                                                  | `+target-lock(?s)`                    |
| `engage(?u ?l ?s ?r)`          | `armed`, `at(?u,?s)`, `*battle`, `*loadout(?u,?l)`, `*direct(?l,?r)`, `window(?r)`, `slot-free` | `+on-station(?r)` (biaya per peran × loadout) |
| `engage-guided(?u ?l ?s ?r)`   | seperti `engage`, tetapi `*guided(?l,?r)` dan `target-lock(?s)`                  | `+on-station(?r)`                     |

- **Grounding dengan fakta statik.** Predikat yang tidak pernah diubah oleh aksi (`adj`, `open`, `aa`, `base`, `battle`, `loadout`, `direct`, `guided`) di-*join* dulu dengan fakta di init. Pada contoh peta, 5744 kombinasi parameter menyusut menjadi 370 aksi ground, misalnya `fly` dari 1792 menjadi 231 karena hanya pasangan sektor yang bertetangga dan terbuka yang dibuat. Predikat persepsi (`window`, `slot-free`, `spotter-alive`) dideklarasikan dinamis, sehingga tetap dicek walaupun tidak ada aksi yang mengubahnya.
- **Forward search.** A\* dengan h-max (*ignore delete lists*, admissible). Sebagai pembanding, UCS dijalankan pada masalah yang sama. Rata-rata dari 45 masalah (15 peta × 3 peran): A\* mengekspansi **37** state, UCS **~2091** (sering mencapai batas 3000). Rencana yang dihasilkan 3–7 langkah.
- **Backward regression.** Aksi harus *relevant* (`add ∩ g ≠ ∅`) dan *consistent* (`del ∩ g = ∅`), lalu `g' = (g − add) ∪ pre`, berhenti saat `g' ⊆ init`. Pencarian memakai A\* mundur dengan h-max dari init, dan subgoal yang memuat pasangan mutex dipangkas.
- **Planning graph.** Level S/A dengan aksi no-op. Mutex aksi mencakup *inconsistent effects*, *interference*, dan *competing needs*; mutex literal memakai *inconsistent support*. Graph dibangun sampai level-off. Level pertama tempat goal muncul tanpa mutex adalah batas bawah panjang rencana (set-level), dan mutex di level akhir dipakai regresi untuk memangkas subgoal mustahil, seperti satu unit berada di dua sektor sekaligus.
- **Eksekusi langkah demi langkah (slide 17).** Komando menjalankan `COMMAND_STEPS_PER_ROUND = 3` langkah tiap akhir ronde. Sebelum tiap langkah, `applicable(s, a)` dicek terhadap state nyata. Bila gagal, misalnya tank sudah direparasi sehingga `window(finisher)` hilang, prekondisi yang hilang dicatat, lalu planner membuat rencana baru dari state saat itu. Bila tidak ada rencana, misi dibatalkan. Seluruh log eksekusi tampil di tab Komando.

---

## Decisions under Uncertainty

Implementasinya ada di [`mdp.py`](mdp.py). Angka di bawah dapat diulang dengan `--mdp-report`.

**Percept → belief → keputusan.** Cuaca sebenarnya (Normal: slip 10%, Badai: slip 25%) diundi saat peta dibuat dan **tidak diberitahukan** ke drone. Ada dua sumber bukti:

- Setiap langkah MDP, drone mengamati apakah gerakannya **meleset**, dengan `P(meleset | w)` dihitung dari model transisi masing-masing cuaca.
- Tombol **Lapor cuaca** memberi laporan dengan akurasi 80%.

Belief di-update dengan aturan Bayes `P(w | e) = P(e | w) P(w) / Σ P(e | w') P(w')`, dengan prior P(Badai) = 0,3. Value iteration dijalankan untuk model transisi tiap cuaca, lalu drone memilih aksi **MEU**:

```
EU(a | e) = Σ_w P(w | e) · Q_w(s, a)          π(s) = argmax_a EU(a | e)
```

Ini adalah aproksimasi QMDP. Overlay per sel menampilkan log Bayes, EU tiap aksi, aksi terbaik bila cuaca diketahui pasti, dan rincian Bellman per cuaca. **Hubungan dengan KB:** goal MDP adalah posisi tank. Bila goal itu berasal dari kontak yang dibagikan rekan, overlay menyebut bahwa goal tersebut hasil pembuktian `Investigate(Drone…)` di KB NPC.

### Analisis desain reward (langkah −0,2 vs −2 di slide)

Slide memakai reward langkah −2 dengan γ = 0,8 untuk dunia 4 state. Di grid 22×16, jarak ke goal bisa ~35 langkah. Tabel berikut adalah hasil rollout π deterministik dari tiap sel bebas, rata-rata 5 peta acak (346 sel per peta):

| Langkah | γ    | Cuaca  | Sampai goal | Bunuh diri ke artileri |
| ------- | ---- | ------ | ----------- | ---------------------- |
| −2      | 0,8  | Normal | 70,1%       | 0%                     |
| −2      | 0,9  | Badai  | 86,8%       | 0,9%                   |
| −2      | 0,95 | Normal | 76,9%       | **23,1%** (79 sel)     |
| −2      | 0,95 | Badai  | 63,0%       | **37,0%** (128 sel)    |
| −0,2    | 0,8  | Normal | 96,4%       | 0%                     |
| −0,2    | 0,9  | Badai  | 98,7%       | 0%                     |
| −0,2    | 0,95 | Normal | 100%        | 0%                     |
| −0,2    | 0,95 | Badai  | 99,0%       | 0%                     |

Dengan langkah −2 dan γ = 0,95, terus berjalan bernilai sekitar −2/(1−γ) = −40, lebih buruk daripada menabrak artileri (−20). Akibatnya, kebijakan optimal untuk sel yang jauh justru "bunuh diri". Ini contoh jebakan desain reward di slide 23: kebijakan itu optimal terhadap reward yang salah. Sebaliknya, dengan γ = 0,8 goal yang jauh hampir tidak bernilai (0,8^35 ≈ 0,0004), sehingga sebagian sel "rabun jauh" dan berputar tanpa sampai goal. Skala −0,2 menjaga rasio langkah : goal : bahaya tetap masuk akal di peta besar, sehingga ≥96% sel sampai goal tanpa bunuh diri.

### Profil performa

- Satu solve value iteration butuh **10–16 ms** (37–69 iterasi, 346 state, 4 aksi).
- Dugaan bahwa VI dihitung ulang tiap tank pindah sel **benar**. Goal adalah posisi tank, sehingga reward ikut pindah dan solusi lama memang tidak berlaku. Pada kode lama, 40 langkah tank menghasilkan 11 miss; cache hanya kena saat tank kembali ke sel lama.
- Biayanya tetap kecil. Simulasi 200 giliran (2 drone mengejar, 2 cuaca, 3 kali gambar per frame) memakan **8,2 ms per giliran**, dibanding tick otomatis 450 ms. Miss hanya terjadi saat tank masuk sel baru (2 cuaca per sel baru); sisanya cache hit (1880 hit, 120 miss).
- Perbaikan yang dilakukan:
  - Kunci cache memakai `map_version`, bukan `hash(tuple(map(tuple, grid)))` yang dihitung beberapa kali per frame. Cache medan memakai cara yang sama.
  - Ukuran cache solusi dinaikkan dari 6 menjadi 16 (8 goal × 2 cuaca, ~3 MB).
  - Kebijakan MEU di-cache per (goal, belief), sehingga layer heatmap hanya digambar ulang saat belief berubah.

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

Program dipecah per materi. Nomor BAGIAN di judul tiap modul berurutan 1–18, mengikuti urutan dependensi.

| Modul                                  | Bagian | Isi                                                                    |
| -------------------------------------- | ------ | ---------------------------------------------------------------------- |
| `world.py`                             | 1–2    | Konstanta medan, pembangkit peta (CSP), posisi awal                    |
| `pathfinding.py`                       | 3–4    | UCS / A\* (`search`), heuristik, line of sight, pola sisir             |
| `ui.py`                                | 5–7    | Warna, `apply_layout`, `Button`, `wrap_text`, overlay inferensi        |
| `battle.py`                            | 8      | `BattleState`, fungsi evaluasi, `MinimaxAgent`, loadout, `Encounter`   |
| `balance.py`                           | 9      | Local search keseimbangan (`--tune`)                                   |
| `npc_local_search.py`                  | 10     | `LocalSearchAgent` (Hill Climbing / Simulated Annealing)               |
| `battle_ui.py`                         | 11     | Pohon pencarian, grafik konvergensi, `BattleController`, `BattleHUD`   |
| `kb.py` + `kb_rules.txt`               | 12     | Parser FOL, CNF, resolusi, pemuat aturan, KB penasihat & NPC           |
| `planner.py`                           | 13     | `Schema`, `ground`, A\*/UCS maju, regresi, `PlanningGraph`             |
| `command.py`                           | 14     | KB komando, domain logistik drone, eksekusi rencana                    |
| `mdp.py`                               | 15     | `MDPSolution`, `WeatherBelief` (Bayes), `BeliefPolicy` (MEU), laporan  |
| `game.py`                              | 16–17  | Drone B dan kelas `Game` (mode eksplorasi)                             |
| `tank_vs_drone_simulasi_ukraina.py`    | 18     | `main()` dan flag CLI                                                  |
| `tests/test_ai.py`                     | –      | Tes unit KB, planner, dan MDP                                          |

Konstanta tata letak (`CELL`, `BOARD_W`, `PANEL_X`, ...) berubah saat jendela di-resize, jadi modul lain membacanya sebagai `ui.CELL` dan bukan `from ui import CELL`.

---

## Konfigurasi

| Konstanta                  | Modul        | Fungsi                                        | Bawaan  |
| -------------------------- | ------------ | --------------------------------------------- | ------- |
| `ROWS`, `COLS`             | `world.py`   | Ukuran grid                                   | 16 × 22 |
| `BASE_CELL`                | `ui.py`      | Ukuran petak minimum (piksel)                 | 30      |
| `BATTLE_TRIGGER_DISTANCE`  | `battle.py`  | Jarak Manhattan pemicu battle                 | 1       |
| `BATTLE_DEFAULT_DEPTH`     | `battle.py`  | Kedalaman pencarian awal                      | 4       |
| `COMMAND_RADAR_RADIUS`     | `kb.py`      | Jangkauan radar penasihat (sel)               | 9       |
| `COMMAND_STEPS_PER_ROUND`  | `command.py` | Langkah rencana bantuan per ronde             | 3       |
| `COMMAND_GRACE_EXTRA`      | `command.py` | Ronde tambahan sebelum bantuan boleh dipanggil | 3       |
| `MDP_STEP_REWARD`          | `mdp.py`     | Reward langkah per biaya terrain              | −0,2    |
| `WEATHER_PRIOR`            | `mdp.py`     | Prior P(Normal), P(Badai)                     | 0,7 / 0,3 |
| `REPORT_ACCURACY`          | `mdp.py`     | Akurasi laporan cuaca                         | 0,8     |
| `AUTO_CHASE_TICK_MS`       | `ui.py`      | Interval gerak otomatis drone                 | 450     |

---

## Catatan

- Bagian visual sengaja disederhanakan dengan primitif Pygame (rect, circle, line).
- Ukuran font tidak ikut membesar saat jendela diperbesar; hanya ruang tata letaknya.
- Pohon pencarian yang digambar diturunkan kedalamannya otomatis bila terlalu lebar agar tetap terbaca.
- Bantuan musuh kini butuh waktu tiba (rencana dieksekusi 3 langkah/ronde). Karena itu `COMMAND_GRACE_EXTRA` diturunkan dari 4 ke 3, lalu dicek dengan simulasi 150 battle per kebijakan: pemain biasa menang 0,47, pemain mahir 0,73, bantuan tiba di 50% / 58% battle. Semuanya masih di pita nyaman `balance.py`. Sebagai pembanding, komando lama menghasilkan 0,43 / 0,66 / 51% / 54%.
