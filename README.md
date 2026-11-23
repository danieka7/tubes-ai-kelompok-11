# Tahap 4 — Logic, Automated Planning, dan Decisions under Uncertainty

Dokumen ini hanya menjelaskan tambahan **Tahap 4** pada simulasi *Tank vs Drone*, yaitu:

| Materi | Topik                                   | Bagian kode                                           |
| ------ | --------------------------------------- | ----------------------------------------------------- |
| 07     | Logic & representasi pengetahuan (FOL)  | BAGIAN 12 – 12.7, aturan di [`kb_rules.txt`](kb_rules.txt) |
| 08     | Automated planning (STRIPS / PDDL)      | BAGIAN 13 – 14, skema aksi di [`drone_domain.txt`](drone_domain.txt) |
| 09     | Decisions under uncertainty (MDP, Bayes, MEU) | BAGIAN 15                                       |

Seluruh kode ada di satu file, [`tank_vs_drone_simulasi_ukraina.py`](tank_vs_drone_simulasi_ukraina.py). Dua file teks di atas dibaca saat program mulai, jadi aturan KB dan skema aksi planner bisa diubah tanpa menyentuh kode.

---

## Menjalankan

```bash
pip install pygame
python tank_vs_drone_simulasi_ukraina.py                # permainan
python tank_vs_drone_simulasi_ukraina.py --mdp-report   # analisis reward & profil MDP (materi 09)
```

`kb_rules.txt` dan `drone_domain.txt` harus berada di folder yang sama dengan file `.py`. Bila salah satunya hilang atau salah sintaks, game memakai versi bawaan di kode (`DEFAULT_RULES` / `DEFAULT_LOGISTICS_SCHEMAS`) dan menampilkan alasannya di layar.

---

## Materi 07 — Logic: Basis Pengetahuan FOL

### Mesin inferensi

Mesin inferensinya generik dan tidak tahu apa-apa tentang game. Isinya:

1. **Parser formula** dengan operator `~`, `&`, `|`, `=>`, `<=>`. Predikat diawali huruf besar, variabel huruf kecil (dianggap berkuantor universal), konstanta berupa nama seperti `R3C4` atau `DroneA`.
2. **Konversi CNF** dalam 4 langkah: hapus `=>`/`<=>`, dorong negasi ke dalam, distribusi `|` atas `&`, lalu kanonisasi klausa.
3. **Unifikasi** dan *rename apart* variabel antar-klausa.
4. **Resolution refutation**: untuk membuktikan `KB ⊨ q`, tambahkan `~q` lalu cari klausa kosong. Strateginya *set-of-support* (hanya turunan dari negasi goal yang dikembangkan), klausa terpendek lebih dulu, dengan batas panjang klausa dan jumlah klausa baru supaya selalu berhenti.

Setiap `ASK` dijalankan dua kali, untuk `q` dan `~q`, sehingga jawabannya tiga kemungkinan:

| Hasil         | Arti                                              |
| ------------- | ------------------------------------------------- |
| **YA**        | `KB ⊨ q`                                          |
| **TIDAK**     | `KB ⊨ ~q`                                         |
| **TIDAK PASTI** | keduanya tidak terbukti; informasi belum cukup  |

### Tiga basis pengetahuan

Aturannya ada di [`kb_rules.txt`](kb_rules.txt), dibagi per bagian `[NAMA_KB]`.

| KB        | Dipakai oleh                     | Pertanyaan (ASK)                   | Fakta persepsi (TELL)                                   |
| --------- | -------------------------------- | ---------------------------------- | ------------------------------------------------------- |
| `ADVISOR` | Penasihat tank (pemain)          | `Safe(sel)`, `Visible(sel)`        | `Passable`, `InRange`, `LineOfSight`, `Close`, roster drone |
| `NPC`     | Drone saling berbagi kontak      | `Investigate(drone)`               | `Spotted`, `Ally`, `Chasing`                            |
| `COMMAND` | Pusat komando musuh (mode battle) | `CallReinforcement`, `Need<Peran>` | 10 proposisi hasil pengamatan pertempuran               |

**KB 1 — Penasihat tank.** Di mode eksplorasi, klik **Tanya Komando**, pilih **Aman?** atau **Terlihat?**, lalu klik sel di peta. KB menjawab apakah sel itu aman dilewati atau terlihat oleh drone. Radar komando hanya menjangkau 9 sel dari tank (`COMMAND_RADAR_RADIUS`); drone di luar radar tidak punya fakta, sehingga jawabannya bisa **TIDAK PASTI**.

**KB 2 — Interaksi NPC.** Bila satu drone melihat tank (`Spotted`), KB menyimpulkan rekannya `Alerted`, dan bila rekan itu belum mengejar maka `Investigate`. Hasil pembuktian ini dipakai sebagai goal pengejaran drone kedua.

**KB 3 — Komando musuh.** Saat battle, komando memutuskan kapan memanggil bantuan dan peran apa yang dibutuhkan (`finisher`, `durable`, `pressure`). Hasilnya menjadi goal planner di materi 08.

### Keputusan desain

- **Bikondisional, bukan aksioma penutup manual.** Contoh:
  `CallReinforcement <=> Outnumbered & ~Busy & GraceOver & SlotFree & QuotaLeft`.
  Konversi CNF menurunkan sendiri klausa arah sebaliknya, jadi KB bisa membuktikan `CallReinforcement` maupun `~CallReinforcement` dari satu kalimat.
- **Tanpa nama drone di aturan.** "Sel tidak terlihat bila *tidak ada* drone yang melihatnya" adalah kuantor universal atas drone, yang tidak bisa ditulis sebagai premis klausa Horn. Solusinya, roster drone ditelusuri berurutan lewat `FirstDrone`, `NextDrone`, `LastDrone` (aturan K5–K7 dan K10–K12). Menambah drone cukup dengan menambah fakta roster; aturannya tidak berubah.
- **Proses inferensi selalu ditampilkan.** Overlay menunjukkan (1) aturan KB, jumlah klausa CNF, dan sumbernya (file atau bawaan kode), (2) fakta TELL, (3) ASK `q` dan `~q` beserta langkah resolusi atau alasan gagal, dan (4) kesimpulan.

---

## Materi 08 — Automated Planning

### Domain logistik drone

Saat KB komando memutuskan memanggil bantuan, planner menyusun rencana agar drone bantuan sampai ke medan tempur. Peta 22×16 dibagi menjadi 4×4 sektor:

- sektor berisi **kamp** → pangkalan (`base`),
- sektor berisi **artileri anti-udara** → tertutup (`aa`), kecuali di-*jam*,
- sektor posisi **tank** → medan tempur (`battle`).

Tujuh skema aksi ditulis di [`drone_domain.txt`](drone_domain.txt):

| Skema                        | Prekondisi (`*` = statik)                                                         | Efek                                        |
| ---------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------- |
| `arm(?u ?b)`                 | `idle(?u)`, `at(?u,?b)`, `*base(?b)`                                              | `+armed(?u)` `-idle(?u)`                    |
| `fly(?u ?a ?b)`              | `armed(?u)`, `at(?u,?a)`, `*adj(?a,?b)`, `*open(?b)`                              | `+at(?u,?b)` `-at(?u,?a)`                   |
| `jam(?s)`                    | `ew-ready`, `*aa(?s)`                                                             | `+suppressed(?s)` `-ew-ready` (biaya 2)     |
| `fly-jammed(?u ?a ?b)`       | `armed`, `at(?u,?a)`, `*adj`, `*aa(?b)`, `suppressed(?b)`                         | pindah sektor                               |
| `designate(?s)`              | `spotter-alive`, `*battle(?s)`                                                    | `+target-lock(?s)`                          |
| `engage(?u ?l ?s ?r)`        | `armed`, `at(?u,?s)`, `*battle`, `*loadout`, `*direct(?l,?r)`, `window(?r)`, `slot-free` | `+on-station(?r)` `+deployed(?u)`    |
| `engage-guided(?u ?l ?s ?r)` | seperti `engage`, tetapi `*guided(?l,?r)` dan `target-lock(?s)`                   | `+on-station(?r)` `+deployed(?u)`           |

Goal-nya `on-station(<peran>)`, dengan peran dari KB 3. Biaya `engage` bergantung pada kecocokan loadout dan peran (`engage-cost`).

Format file domain:

```
action <nama> ?param:tipe ...
  pre:  fakta ; fakta
  add:  fakta ; fakta
  del:  fakta ; fakta      (opsional)
  cost: angka | engage-cost (opsional, bawaan 1)
  doc:  keterangan         (opsional)
```

Nama aksi `engage` dan `engage-guided` jangan diubah, karena kode komando mengenalinya.

### Algoritma

- **Grounding dengan fakta statik.** Predikat yang tidak pernah diubah aksi (`adj`, `open`, `aa`, `base`, `battle`, `loadout`, `direct`, `guided`) di-*join* dulu dengan fakta awal, sehingga hanya kombinasi parameter yang mungkin yang dibuat. Pada contoh peta, 5744 kombinasi menyusut menjadi 370 aksi ground. Predikat persepsi (`window`, `slot-free`, `spotter-alive`) tetap dicek walau tidak ada aksi yang mengubahnya.
- **Forward search.** A\* dengan heuristik **h-max** (*ignore delete lists*, admissible sehingga rencana optimal). UCS dijalankan pada masalah yang sama sebagai pembanding: rata-rata A\* mengekspansi puluhan state, UCS ribuan.
- **Backward regression.** Aksi harus *relevant* (`add ∩ g ≠ ∅`) dan *consistent* (`del ∩ g = ∅`), lalu `g' = (g − add) ∪ pre`, berhenti saat `g' ⊆ init`.
- **Planning graph.** Level S/A dengan aksi *no-op*. Mutex aksi: *inconsistent effects*, *interference*, *competing needs*. Mutex literal: *inconsistent support*. Graph dibangun sampai *level-off*; level pertama goal muncul tanpa mutex menjadi batas bawah panjang rencana, dan mutex dipakai regresi untuk memangkas subgoal mustahil (misalnya satu unit di dua sektor sekaligus).
- **Eksekusi dan replanning.** Komando menjalankan 3 langkah rencana tiap akhir ronde (`COMMAND_STEPS_PER_ROUND`). Sebelum tiap langkah, prekondisi dicek terhadap state nyata. Bila gagal (misalnya tank sudah reparasi sehingga `window(finisher)` hilang), prekondisi yang hilang dicatat dan planner membuat rencana baru dari state saat itu. Bila tidak ada rencana, misi dibatalkan.

Semua hasil (aksi ground, rencana A\* vs UCS, regresi, planning graph, log eksekusi) tampil di tab **Komando** pada mode battle.

---

## Materi 09 — Decisions under Uncertainty

### Model MDP

Pada A\*/UCS drone membuat *rencana* tetap. Pada mode **MDP**, drone memakai *kebijakan* π(s) yang tetap valid walau gerakannya meleset.

| Komponen     | Isi                                                                              |
| ------------ | -------------------------------------------------------------------------------- |
| State        | Sel peta. Posisi tank (goal) dan sel artileri adalah terminal.                   |
| Aksi         | Atas, Kanan, Kiri, Bawah                                                         |
| Transisi     | Berhasil dengan peluang `1 − slip`, meleset ke dua sisi tegak lurus masing-masing `slip/2`. Keluar peta = diam. |
| Reward       | Goal +10, artileri −20, selain itu −0,2 × biaya terrain                          |
| Discount γ   | 0,8 / 0,9 / 0,95 (bawaan 0,9)                                                    |

Persamaan Bellman dan value iteration:

```
Q(s,a) = Σ_s' P(s'|s,a) [ R(s') + γ V(s') ]
V(s)   = max_a Q(s,a)
```

Mulai dari V₀ = 0, iterasi berhenti saat perubahan terbesar < 10⁻⁴.

### Belief cuaca (Bayes) dan keputusan MEU

Cuaca sebenarnya diundi saat peta dibuat dan **disembunyikan** dari drone:

| Cuaca  | Peluang slip | Prior |
| ------ | ------------ | ----- |
| Normal | 10%          | 0,7   |
| Badai  | 25%          | 0,3   |

Drone memperbarui belief dari dua jenis bukti:

1. Setiap langkah, apakah gerakannya **meleset** atau tidak, dengan `P(meleset | cuaca)` dari model transisi.
2. Tombol **Lapor cuaca**, laporan dengan akurasi 80%.

```
P(w | e) = P(e | w) P(w) / Σ_w' P(e | w') P(w')
```

Value iteration dijalankan untuk tiap cuaca, lalu drone memilih aksi dengan *Maximum Expected Utility*:

```
EU(a | e) = Σ_w P(w | e) · Q_w(s, a)        π(s) = argmax_a EU(a | e)
```

Ini aproksimasi QMDP. Bila goal drone berasal dari kontak yang dibagikan rekan, overlay menyebut bahwa goal itu hasil pembuktian `Investigate(...)` di KB NPC, jadi materi 07 dan 09 saling terhubung.

### Kenapa reward langkah −0,2, bukan −2 seperti di slide?

Slide memakai −2 dengan γ = 0,8 untuk dunia 4 state. Di grid 22×16, jarak ke goal bisa ~35 langkah. Dengan −2 dan γ = 0,95, terus berjalan bernilai sekitar −2/(1−γ) = −40, lebih buruk daripada menabrak artileri (−20). Akibatnya kebijakan optimal untuk sel yang jauh justru "bunuh diri" (23–37% sel). Dengan −0,2, ≥96% sel sampai goal tanpa bunuh diri. Angka lengkapnya bisa dilihat dengan `--mdp-report`.

### Kontrol mode MDP

| Input                              | Fungsi                                               |
| ---------------------------------- | ---------------------------------------------------- |
| Tombol **MDP**                     | Drone mengejar dengan kebijakan MEU                  |
| **gamma 0.8 / 0.9 / 0.95**         | Mengganti discount                                   |
| **Nyata: Normal/Badai**            | Mengganti cuaca sebenarnya (drone tidak diberi tahu) |
| **Lapor cuaca**                    | Laporan cuaca 80% akurat + update Bayes              |
| **Tampil: Panah / Angka V / Mati** | Tampilan kebijakan atau V(s) di peta                 |
| **< Iterasi >** / **Akhir**        | Menelusuri iterasi value iteration                   |
| Klik sel                           | Overlay log Bayes, EU tiap aksi, dan Bellman per cuaca |

Satu solve value iteration butuh sekitar 10–16 ms. Solusi di-cache per (goal, cuaca), dan kebijakan MEU di-cache per (goal, belief), sehingga game tetap lancar walau VI dihitung ulang setiap tank pindah sel.

---

## Konfigurasi Tahap 4

| Konstanta                 | Fungsi                                          | Bawaan    |
| ------------------------- | ----------------------------------------------- | --------- |
| `COMMAND_RADAR_RADIUS`    | Jangkauan radar penasihat (sel)                 | 9         |
| `COMMAND_STEPS_PER_ROUND` | Langkah rencana bantuan per ronde               | 3         |
| `COMMAND_GRACE_EXTRA`     | Ronde tambahan sebelum bantuan boleh dipanggil  | 3         |
| `MDP_STEP_REWARD`         | Reward langkah per biaya terrain                | −0,2      |
| `MDP_GOAL_REWARD` / `MDP_HAZARD_REWARD` | Reward goal / artileri            | +10 / −20 |
| `WEATHER_PRIOR`           | Prior P(Normal), P(Badai)                       | 0,7 / 0,3 |
| `REPORT_ACCURACY`         | Akurasi laporan cuaca                           | 0,8       |
