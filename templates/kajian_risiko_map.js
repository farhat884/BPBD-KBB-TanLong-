// ================================================================
// KAJIAN RISIKO BENCANA (mode peta)
// ----------------------------------------------------------------
// Blok ini disisipkan oleh app.py (generate_map) ke dalam <script>
// peta, lalu tanda-tanda KAJIAN (yang diapit garis bawah ganda) di
// bawah diganti JSON dari kajian_risiko.py.
//
// Tingkat risiko TIDAK dihitung di sini -- semuanya kelas
// (Rendah/Sedang/Tinggi) yang dibaca langsung dari tabulasi BPBD.
// Yang dilakukan blok ini hanya menampilkan: warna peta, tooltip,
// legenda, dan dropdown pilihan jenis bencana.
//
// Kunci desa = "<kunci kecamatan>|<kunci desa>" (gabungan) karena ada
// nama desa yang sama di kecamatan berbeda (mis. Mekarjaya, Cipada).
// ================================================================

var kajianHazards = __KAJIAN_HAZARDS_JSON__;
var kajianKab = __KAJIAN_KAB_JSON__;
var kajianKec = __KAJIAN_KEC_JSON__;
var kajianDesa = __KAJIAN_DESA_JSON__;
var kajianTersedia = __KAJIAN_TERSEDIA__;
var kajianWarna = __KAJIAN_WARNA_JSON__;
var kajianHazardAktif = '__KAJIAN_DEFAULT_HAZARD__';

// Kunci kecamatan yang sedang dibuka (tampilan desa); null = tampilan
// seluruh kecamatan. Dipakai supaya angka di legenda desa cuma
// menghitung desa di kecamatan yang sedang dilihat.
var kajianKecTerbuka = null;

// Warna TEKS per kelas (lebih gelap dari warna isi peta supaya kebaca
// di latar putih tooltip).
var kajianWarnaTeks = { Rendah: '#15803d', Sedang: '#b45309', Tinggi: '#b91c1c' };


function kajianEsc(s) {
    return String(s === null || s === undefined ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function kajianAngka(n, maxDesimal) {
    return Number(n).toLocaleString('id-ID', { maximumFractionDigits: maxDesimal === undefined ? 0 : maxDesimal });
}

// Nilai kerugian di tabulasi berupa JUTA rupiah -> tampilkan satuan yang enak dibaca.
function kajianRupiah(juta) {
    var n = Number(juta);
    if (n >= 1000000) return 'Rp ' + kajianAngka(n / 1000000, 2) + ' triliun';
    if (n >= 1000) return 'Rp ' + kajianAngka(n / 1000, 2) + ' miliar';
    return 'Rp ' + kajianAngka(n, 2) + ' juta';
}

function kajianHazardMeta(id) {
    for (var i = 0; i < kajianHazards.length; i++) {
        if (kajianHazards[i].id === id) return kajianHazards[i];
    }
    return kajianHazards[0];
}

// Label pendek untuk judul (buang keterangan dalam kurung).
function kajianLabelPendek(hz) {
    return String(hz.label).split(' (')[0];
}

function kajianKeyDesa(properties) {
    var namaDesa =
        properties.village ||
        properties.nama_desa ||
        properties.desa ||
        properties.nm_desa ||
        properties.DESA ||
        properties.NAMOBJ;
    var namaKec =
        properties.nm_kecamatan ||
        properties.Kecamatan ||
        properties.kecamatan;
    return cleanNameJS(namaKec) + '|' + cleanNameJS(namaDesa);
}

function kajianRec(level, key) {
    var store = (level === 'kec') ? kajianKec : kajianDesa;
    var e = store[key];
    if (!e || !e.h) return null;
    return e.h[kajianHazardAktif] || null;
}

function kajianFillColor(level, key) {
    var rec = kajianRec(level, key);
    if (rec && kajianWarna[rec.r]) return kajianWarna[rec.r];
    return kajianWarna._none;
}

function kajianKelasTeks(kelas) {
    if (!kelas) return '-';
    return '<b style="color:' + (kajianWarnaTeks[kelas] || '#334155') + ';">' + kajianEsc(kelas) + '</b>';
}

function kajianTooltipHtml(level, key) {
    var hz = kajianHazardMeta(kajianHazardAktif);
    var store = (level === 'kec') ? kajianKec : kajianDesa;
    var e = store[key] || {};
    var nama = kajianEsc(String(e.n || '').toUpperCase());

    var html = '<div style="font-family:Arial, sans-serif; min-width:200px; padding:4px;">';
    html += '<b style="font-size:13px;color:#2c3e50;">' + (level === 'kec' ? 'KEC. ' : 'DESA ') + nama + '</b>';
    if (level === 'desa' && e.k) {
        html += '<div style="font-size:10px;color:#64748b;">Kec. ' + kajianEsc(e.k) + '</div>';
    }
    html += '<hr style="margin:4px 0;border:0;border-top:1px solid #ccc;">';
    html += hz.icon + ' <b>' + kajianEsc(hz.label) + '</b><br>';

    if (!kajianTersedia) {
        html += '<span style="font-size:10px;color:#7f8c8d;"><i>Data matriks kajian risiko belum tersedia.</i></span>';
        return html + '</div>';
    }

    var rec = kajianRec(level, key);
    if (!rec) {
        html += '<span style="font-size:11px;color:#64748b;"><i>Tidak tercatat dalam matriks kajian risiko untuk bencana ini.</i></span>';
        return html + '</div>';
    }

    html += 'Tingkat Risiko: <b style="font-size:13px;color:' + (kajianWarnaTeks[rec.r] || '#334155') + ';">' + kajianEsc(rec.r) + '</b><br>';
    html += '<span style="font-size:10px;color:#475569;">' +
        'Bahaya: ' + kajianKelasTeks(rec.b) + ' &middot; ' +
        'Kerentanan: ' + kajianKelasTeks(rec.v) + ' &middot; ' +
        'Kapasitas: ' + kajianKelasTeks(rec.c) +
        '</span><br>';
    if (rec.p !== undefined) {
        html += '👥 Penduduk Terpapar: <b>' + kajianAngka(rec.p) + '</b> jiwa<br>';
    }
    if (rec.k !== undefined) {
        html += '💰 Potensi Kerugian: <b>' + kajianRupiah(rec.k) + '</b><br>';
    }
    if (rec.lr !== undefined) {
        html += '🗺️ Luas Area Berisiko: <b>' + kajianAngka(rec.lr, 2) + '</b> ha';
    }
    return html + '</div>';
}


// ---------------- LEGENDA ----------------

function kajianHitung(level) {
    var hasil = { Rendah: 0, Sedang: 0, Tinggi: 0, none: 0, total: 0 };
    var store = (level === 'kec') ? kajianKec : kajianDesa;
    for (var key in store) {
        if (level === 'desa' && kajianKecTerbuka && key.indexOf(kajianKecTerbuka + '|') !== 0) continue;
        var rec = kajianRec(level, key);
        hasil.total++;
        if (rec && hasil[rec.r] !== undefined) hasil[rec.r]++;
        else hasil.none++;
    }
    return hasil;
}

function kajianLegendHtml(level) {
    var hitung = kajianHitung(level);
    var satuan = (level === 'kec') ? 'kecamatan' : 'desa';
    var baris = [
        ['Rendah', kajianWarna.Rendah, 'Rendah'],
        ['Sedang', kajianWarna.Sedang, 'Sedang'],
        ['Tinggi', kajianWarna.Tinggi, 'Tinggi'],
        ['none', kajianWarna._none, 'Tidak tercatat']
    ];
    var html = '';

    if (level === 'kec') {
        var recKab = kajianKab[kajianHazardAktif];
        html += '<div style="font-size:11px;margin-bottom:6px;">Tingkat Kab. Bandung Barat: ' +
            (recKab ? '<b style="color:' + (recKab.r === 'Sedang' ? '#fbbf24' : (recKab.r === 'Tinggi' ? '#f87171' : '#4ade80')) + ';">' + kajianEsc(recKab.r) + '</b>' : '<i>-</i>') +
            '</div>';
    }

    for (var i = 0; i < baris.length; i++) {
        html += '<div style="display:flex;align-items:center;gap:6px;font-size:11px;margin:2px 0;">' +
            '<span style="display:inline-block;width:14px;height:14px;border-radius:3px;border:1px solid rgba(255,255,255,0.5);background:' + baris[i][1] + ';"></span>' +
            '<span>' + baris[i][2] + '</span>' +
            '<span style="margin-left:auto;color:#cbd5e1;">' + hitung[baris[i][0]] + ' ' + satuan + '</span>' +
            '</div>';
    }
    if (hitung.none > 0 && kajianTersedia) {
        html += '<div style="font-size:9px;color:#94a3b8;margin-top:4px;">Abu-abu = wilayah ini tidak tercatat di matriks untuk bencana yang dipilih.</div>';
    }
    if (!kajianTersedia) {
        html += '<div style="font-size:10px;color:#fca5a5;margin-top:4px;">Data matriks kajian risiko belum tersedia.</div>';
    }
    return html;
}

// Tampilkan legenda kategori (Rendah/Sedang/Tinggi) di mode kajian
// dan sembunyikan gradasi + label min/max bawaan; kebalikannya di mode lain.
function kajianAturLegenda(mode) {
    var aktif = (mode === 'kajian_risiko');
    var levels = [['kec', 'legend-kec'], ['desa', 'legend-desa']];

    for (var i = 0; i < levels.length; i++) {
        var level = levels[i][0];
        var prefix = levels[i][1];
        var bar = document.getElementById(prefix + '-bar-wrap');
        var minmax = document.getElementById(prefix + '-minmax');
        var blok = document.getElementById(prefix + '-kajian');

        if (bar) bar.style.display = aktif ? 'none' : 'block';
        if (minmax) minmax.style.display = aktif ? 'none' : 'flex';
        if (blok) {
            blok.style.display = aktif ? 'block' : 'none';
            if (aktif) blok.innerHTML = kajianLegendHtml(level);
        }
    }
}

// Judul & subjudul legenda untuk mode kajian (dipanggil dari updateLegendText).
function kajianJudulLegenda() {
    var hz = kajianHazardMeta(kajianHazardAktif);
    return {
        judul: 'Tingkat Risiko ' + hz.icon + ' ' + kajianEsc(kajianLabelPendek(hz)),
        subjudul: 'Klasifikasi langsung dari matriks Kajian Risiko Bencana KBB 2026&ndash;2030'
    };
}


// ---------------- DROPDOWN ----------------

function setKajianHazard(id) {
    kajianHazardAktif = id;
    if (currentColorMode === 'kajian_risiko') {
        applyColorMode('kajian_risiko');
    }
}

function kajianAturDropdown(mode) {
    var picker = document.getElementById('kajian-picker');
    if (picker) picker.style.display = (mode === 'kajian_risiko') ? 'flex' : 'none';
}
