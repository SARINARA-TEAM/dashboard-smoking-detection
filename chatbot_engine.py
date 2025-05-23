import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

# custom stopwords
custom_stopwords = {
    "alo", "dok", "halo", "dokter",
    "selamat", "pagi" "siang", "sore", "malam",
    "terimakasih", "makasih", "hunnie",
    "alodokter", "hai", "salam", "hay", "doc",
    "Assalamualaikum", "permisi", "docter", "mr", "mrs", "Aslamualakum", "ass",
    "hallo", "Assalamuallaikum", "mlm", "tanya", "assalammualaikum", "dear", "salam",
    "Aslmkum.wr.wb", "maaf", "hello", "perkenalkan", "hallow", "alodok", "slmt", "mlm",
    "hey", "hormat", "nama", "mohon", "dimohon", "jawab", "pertanyaan", "jawaban", "salam", "hormat",
    "Asalamualaikum", "doktor", "haloo", "pencerahan", "pencerahannya", "Assalamu a'laikum", "aslmkm",
    "Assallamualaikum", "Asslmkm", "Assamualaikum", "perkenalkan", "waalaikumslaam", "astari", "eldison",
    "khoirul", "halim", "dyah", "adhe", "dilly", "frengky", "agus", "usman", "taqiyyah", "rin", "rizki", "amirul",
    "arini", "rin", "aldrian", "fikry", "zahirah", "robby", "rendi", "fitranta", "adam","zahirah", "yannie", "ayundya",
    "maulana", "ricky", "grivaly", "aisyah", "awi", "nadnad", "kharisma", "ahmed", "david", "edo", "arhy", "stevander",
    "reyn", "ibnu", "fajar", "muhardy", "mario", "zainatun", "icha", "rendi", "fitranta", "yusufa", "lina", "wahyu",
    "arif", "raihan", "nijam", "rakha", "yuniarselaa", "sukma", "ezrael", "saeful", "thoriq", "edho", "alvirna", "erik",
    "irfan", "fransisa", "eidrus", "samuel", "nurikah", "aldialam2016", "beni", "nurul", "munir", "rudy", "dhimas", "denny",
    "pejok", "pratama", "muhammad", "rhyo", "andi", "clara", "andreas", "fitra", "nadhifa", "dessy", "firman", "feetrc", "ahmad",
    "daniel", "ardiansyah", "iesti", "yana", "rizky", "rindi", "putri", "danie", "aditiya", "aris", "rizky", "rozab", "hidayat",
    "robo", "bram", "popoye", "sharie", "maman", "ottong", "mayang", "christian", "azwar", "yaman", "agus", "ady", "egi", "ivan",
    "hasto", "rony", "aya", "rheny", "andi", "yansen", "fauzi","dr.",

    'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir',
    'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara',
    'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal',
    'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan',
    'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
    'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini',
    'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja',
    'belakang', 'belakangan', 'belum', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada',
    'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun',
    'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya',
    'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan',
    'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula',
    'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut',
    'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa',
    'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat',
    'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup',
    'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang',
    'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri',
    'diakhirinya', 'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya',
    'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya',
    'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan',
    'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan',
    'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta',
    'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 'dini',
    'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan', 'diperkirakan', 'diperlihatkan',
    'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan', 'dipunyai', 'diri', 'dirinya',
    'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah', 'ditambahkan',
    'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk',
    'ditunjuki', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya',
    'diucapkan', 'diucapkannya', 'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak',
    'enggaknya', 'entah', 'entahlah', 'guna', 'gunakan', 'hal', 'hampir', 'hanya', 'hanyalah',
    'hari', 'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia',
    'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin',
    'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah',
    'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas',
    'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru',
    'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan',
    'kapan', 'kapankah', 'kapanpun', 'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah',
    'katanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan',
    'kelihatan', 'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan',
    'kemungkinannya', 'kenapa', 'kepada', 'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya',
    'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya', 'kita',
    'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lama',
    'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam', 'maka', 'makanya',
    'makin', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa',
    'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun',
    'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang', 'memastikan', 'memberi',
    'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan',
    'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan',
    'mempersoalkan', 'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan', 'menaiki',
    'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti', 'menantikan', 'menanya', 'menanyai',
    'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi', 'mendatangkan',
    'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai',
    'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya',
    'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya',
    'mengungkapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki',
    'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan', 'menyampaikan', 'menyangkut',
    'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah',
    'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip', 'misal',
    'misalkan', 'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'nah',
    'naik', 'namun', 'nanti', 'nantinya', 'nyaris', 'nyatanya', 'oleh', 'olehnya', 'pada', 'padahal',
    'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti', 'pastilah', 'penting', 'pentingnya', 'per', 'percuma',
    'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan', 'pertama', 'pertama-tama', 'pertanyaan',
    'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya', 'rata',
    'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai',
    'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se', 'sebab',
    'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya',
    'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya',
    'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara',
    'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'seenaknya', 'segala',
    'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah',
    'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang',
    'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya',
    'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya',
    'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih',
    'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua',
    'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seolah-olah', 'seorang',
    'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya', 'sepihak',
    'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali',
    'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
    'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi',
    'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya',
    'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak',
    'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan',
    'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah',
    'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri',
    'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya',
    'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut',
    'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba',
    'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk', 'turut', 'tutur',
    'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk',
    'usah', 'usai', 'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong', 'yaitu',
    'yakin', 'yakni', 'yang'
}

custom_stopwords_phrases = {
    "alo dok", "halo dok", "halo dokter", "alo dokter",
    "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "mau tanya", "saya mau tanya", "dok mau tanya",
    "terimakasih", "terima kasih", "makasih", "mohon info",
    "mohon penjelasan", "izin bertanya", "saya ingin bertanya",
    "alo", "dok", "dokter", "halo", "hai", "hunnie","ingin tanya",
    "di alodokter", "terima kasih telah bertanya ke alodokter",
    "terima kasih atas pertanyaannya", "Pagi bapak/ ibu dokter", "dok saya ingin konsultasi",
    "Alodokter dok", "Saya mau konsultasi dok", "saya mau tanya", "aya mau tanya dok",
    "sy mau tanya", "slamat siang", "salam dok","salam sehat", "aya Angga umur 29 thn"
    "saya mau nanya", "Assalam wrb dok", "Ass Dokter", "Assalamuaakaikum dok", "Maaf dok nanya", "terima kasih ya sudah bertanya di Alodokter"
    "Maaf dok nanya", "Satu hal lg dok", "Mau tanya sok", "bersumber dari artikel", "saya ady", "usia 25 thn", "terimakasih telah bertanya ke Alodokter",
    "misi dok", "Mau tanya lagi dok", "Minta pendapat dokk", "saya syafrudin", "Perkenalkan saya richard", "saya mau tanya dan pendapat"
    "Saya mau ty ma Mr/Mrs. Dokter nich", "nama saya Debby", "nama saya", "saya mau bertanya", "Slmat mlm", "dok saya mau banyak",
    "Assalamualaikum wr.wb", "saya hanya ingin bertanya", "mt mlm", "mf mngganggu lg", "aq ingn brtax", "Alodokter met sore", "saya mau tanya ini",
    "minta solusinya", "nama saya ganeva", "Assalamualaikum.wr wb", "kasih tau dong", "dokter yang mulia", "As salaamu Alaikum",
    "maaf dok", "Dokter Alodokter.com", "nm sy fahry", "umur 32thun", "Tlong bntuannya ya", "saya mau bertanya menurut dokter",
    "mohon informasinya", "met malam", "Ass maaf ganggu waktu nya", "maaf saya mau nanya", "hlo dok", "saya firman", "umur 30 th", "mu tanya",
    "sy ingin bertanya", "sy mau tanya", "salam sejahteradok", "saya alifi umur 20", "selamat idulfitri", "sya mau bertanya", "nama saya iqbal umur 21thn",
    "nama saya", "jason umur 14 tahun", "saya mo tanya", "hi dokter", "Sy dari Malaysia", "Sy dari", "Sebelum itu maaf ya saya gk bisa tutur dalam bahasa indonesia secara perfect",
    "Saya ada 2 soalan", "maaf ganggu", "sya mau tanya", "Selamat malam doktet", "pahlawan banyak jasa", "saya yuni", "hay dok", "saya wahyu", "hey dok slamat sore", "perkenalkan saya rully laki2 usia 27 tahun",
    "Saya annisa lingga utami", "saya apri", "saya ingin bertanya kepada dokter", "hi dok", "Aslmkum.wr.wb slamat pagi", "maaf sebelumnya", "saya inggin bertanya", "hay bu dok", "menurut dokter", "saya mo bertanya",
    "Saya Hasti 26th", "saya mau menanyakan", "saya ingin menanyakan", "mohon jawaban nya", "nama saya aji", "Saya dari Garut, umur saya 20 tahun", "saya dari", "mau tanya nii", "saya yorky", "ingin bertanya", "hallow dok",
    "menurut dokter", "hy dok", "mau tny nih", "mohon maaf", "mohon maaf mengganggu", "malem dok", "malem dokter","saya mau tanya nih", "slmt mlm", "Salam untuk dokter YTH", "dengan hormat", "saya nano dri bekasi", "mohon pencerahan nya",
    "begini dokter", "menurut dokter", "tolong jawab pertanyaan saya yah", "saya mau nanyak", "Saya I Gede Aries Suartama", "mau bertanya", "saya mw tanya nih", "saya mau nnya", "Saya Nova umur 27 th", "nama saya ardhi umur23", "maaf sebelumnya",
    "saya mau tnya", "nama saya habib", "Selamat pagi menjelang siang", "aya Arif Setiawan Umur 21", "yang saya ingin tanyakan", "sy mau tanya lg", "met pagi", "saya dwi dari jogja", "mau tanya", "maaf mau tanya", "maaf mengganggu aktivitas",
    "Saya Ezar", "nama saya zidane", "izin dokter", "met siang", "met pagi", "met malem", "met sore", "nama saya zahra", "saya jaki umur saya 19", "mw tnya ni", "saya yuliani", "Namaku rani,umurku 17 tahun", "Saya Rana umur 16 thn", "saya samsul umur 22",
    "maaf dokter", "Saya nandang usia 49, Bogor", "saya wahyu", "sayavmau bertanya", "Nama saya dwi umur 24th", "saya kevin", " yang saya mau tanyakan", "Selamat malamSaya mau tanya", "saya ada pertanyaan", "Saya Viko Andreas berusia 22 Tahun", "Saya fauzi",
    "mw tanya", "saya jamal 27 tahun", "Saya Suleman", "Nama saya Billy", "usia 25 tahun", "aku Hasanah umur 30 thn", "sy putra 20 thun", "ingin menanyakan", "saya ingin menanyakan", "saya Anjar", "ijin bertanya", "izin bertanya", "maaf saya mau bertanya", "malam dokmaaf mau tanya",
    "ijn tanya", "sy ingin tanya", "saya ingin konsultasi", "alo bambang", "alo balpin", "alo bajoll", "alo aurel", "alo stari", "alo ardan", "alo anggi", "alo alya", "alo aliee", "alo aldi", "alo al", "alo aisyah", "alo adi", "alo abdul", "alo abd", "aldy valentino", "agness maureen",
    "Aditya Pratama", "adhi pasha", "adam achmad hamdallah", "alo bayu", "allo bella", "alo berlian", "alo billy", "alo cindy", "alo dava", "alo dedi", "alo devi", "alo dicky", "alo dinda", "alo dini", "alo dito", "alo dwi", "alo eka", "alo elly", "alo endah", "alo erik", "alo erna",
    "alo eva", "alo fadil", "alo farah", "alo fikri", "alo fina", "alo gita", "alo hafidz", "alo hana", "alo haris", "alo ika", "alo indah", "alo indra", "alo ira", "alo irma", "alo iwan", "alo jaya", "alo jenny", "alo johan", "alo joni", "alo kartika", "alo kevin", "alo lia", "alo lilis", "alo lintang",
    "alo lucky", "alo lutfi", "alo maria", "alo mega", "alo melati", "alo miko", "alo mira", "alo nadia", "alo nanda", "alo nia", "alo nina", "alo novi", "alo nura", "alo putri", "alo rafi", "alo rani", "alo riko", "alo rina", "alo rizal", "alo rudy", "alo sari", "alo sinta", "alo siti", "alo syifa", "alo tika",
    "alo toni", "alo vina", "alo wahyu", "alo wati", "alo yani", "alo yuda", "alo yuli", "alo zaki", "alo bertha", "terimakasih atas pertanyaannya", "saya dr. Tirtawati Wijaya", "terima kasih telah bertanya", "terima kasih telah bertanya di Alodokter", "terima kasih sudah bertanya kepada alodokter", "terima kasih sudah bertanya",
    "Saya dr. Ainul", "Terima kasih atas pertanyaannya untuk Alodokter", "Terima kasih atas pertanyaannya", "terimakasih sudah menghubungi alodokter", "terimakasih sudah menghubungi", "terimakasih ataspertanyaannya", "terimakasih sudah bertanya di AlodokterSaya", "terimakasih sudah bertanya", "Terima kasih telah bertanya kepada kami",
    "Terima kasih telah bertanya di alodokter", "Alo Saptaning", "Terima kasih sudah bertanya ke Alodokter", "Alo Shane", "terima kasih sudah bertanya", "Alo dikimaulana", "alo nathaael", "halo adam", "terimakasih sudah menghubungi alodokter", "alo acha", "Terima kasih telah bertanya ke Alodokter", "Terima kasih telah bertanya", "alo nafian",
    "alo eldison", "alo cachink", "terima kasih ya sudah bertanya", "halo nindya", "alo rosaria", "halo rama", "alo chelsy", "alo fajri", " terimakasih atas pertanyaan anda di Alodokter", " terimakasih atas pertanyaan anda", "alo restu", "alo denny", "alo muhammad", "hai elya", "alo habibi",
    " terimakasih telah bertanya di Alodokter", " terimakasih telah bertanya", "alo rohmat", "salam alodokter", "alo kris", "terima kasih sudah bertanya pada Alodokter", "Selamat malam. Terimakasih sudah bertanya ke alodokter", "alo lydya", "alo haryanto", "Terima kasih telah berkonsultasi ke Alodokter", "Terima kasih telah berkonsultasi",
    "halofauzi nugraha", " terima kasih telah bertanya pada Alodokter", " terima kasih telah bertanya", "hallo krisna", "terima kasih telah berkonsultasi dengan kami di web Alodokter.com", "terima kasih telah berkonsultasi dengan kami", "alo nur", "Perkenalkan saya dr.Rio", "kan menjawab pertanyaan yang anda berikan", "salam alodokter", "alo muslikh", "alo sifak", "alo ilham", "alo dhopi", "alo bajoll", "alo destina", " alo fauzanrez", "hello rinawati",
    "alo al", "Terimakasih telah menggunakan layanan konsultasi Alodokter", "Terimakasih telah menggunakan layanan konsultasi", "alo patma", "alo setiyo budi", "ao anggie", "hallo anita sari", "alo lala3030", "terima kasih sudah bertanya", "alo rengga", "alo heru", "alo mario", "alo proo", "alo iskandar azman tsuna", "alo ar sungli", "hallo sdr. deli ayu", "alo simon", "halo jihan", "alo lenny", "selamat pagi mason", "selamat pagi kenny", "alo bobby",
    "alo mutiara", "selamat malam samsul", "alo mr.xing", "alo blank", "alo stanley", "selamat pagi amara", "selamat siang bagas", "alo stanley", "alo viky", "alo handy", "alo juan", "Selamat sore, Fifie Gabriel, terima kasih ya sudah bertanya di Alodokter", "alo siska", "halo rizal mantovani", "alo astari", "halo fiki steeven", "selamat malam hiroshi hamada", "hai rejeki", "hallo miya", "hallo rangga", "halo stefanus renaldi", "hai ibnuwepe", "halo yanuar97",
    "hai gen", "halo jhoni", "halo veri", "hai stefanus", "hai annisa90", "halo ihzan", "halo stefanus", "halo shafasalsabila", "halo muhamad fikri", "hai muhammadeustass", "hai wil4697", "hai latiffatul", "halo kelvinjunior", "halo selamat siang tommi", "halo fika", "halo sumsum", "hai kirun", "halo anastasiarenataa", "halo muhammad rifqy", "halo salmbar rewa", "halo riri", "halo maiden", "halo gunadi putra winata", "w m anggoro", "hai nissa", "halo tn syahdikabobby",
    "halo tn fauzi gunawan", "halo arsyad", "hai rafly", "hallo rahmadramadhan", "faisal yang baik", "halo tn beni", "halo afik affakih", "halo ray", "halo bapak tampan", "hai fredy", "hallo reinhard", "halo defani", "halo karina", "delfa yang baik", "halo lintang", "halo alfin27", "hai alfin", "Wa'alaikumsalam wr.wb Oghy", "hai mannuea", "hai rodi", "hai sigit", "hai skizofrenia", "halo dodo240593", "halo gio", "halo farhan gani", "halo ahmad", "halo arya", "hai yuliansor",
    "halo arif", "hai venni", "hai tiara", "halo dian", "halo reza", "hai widi", "resky yang baik", "hai hunnie", "halo stefanus01", "hai anastacia", "halo ariprass", "gunadi yang baik", "hai syadkha", "ratu yang baik", "kusnadi yang baik", "halo hendriawam", "halo anis purwanti", "habib yang baik", "waalaikumsalam wr,wb nur", "hai fadil", "halo monifa29", "ali aulia1524", "halo raiessa", "hallo bunda iyar", "hai arie muhammad", "ardhi fort minor", "halo de javu",
    "halo jahotman", "halo alim", "halo tuwing", "hai alim", "halo tuwing", "hallo alfie kusumah", "halo damaya", "halo nova lyana aice", "halo saifudin", "hai hilman", "halo astri", "halo den fauzi", "halo linda", "halo bayu", "halo ucok", "hai i gede", "hai edryan", "hai indah yuliani", "halo yoga sepriansyah", "hai sak hadi", "selamat malam temito", "hai androin", "halo ribka", "halo", "hai rizal", "hai", "halo maria", "halo muzakhir", "halo zeera", "hai",
    "selamat malam rico pratama", "halo", "hai iko roma dani", "halo", "halo", "halo nur ichsan", "hai", "salam", "halo rana", "xananta yang baik", "halo siscaquinn", "halo jimen", "dan 1 lagi dok mau share pertanyaan apa yang menyebabkan paru paru dengan kondisi dada berbentuk tong dan takipnue", "hai", "salam", "halo mbak angel", "halo puput", "dear indra", "hai robby", "halo fakhri", "hai", "terima kasih fiqih atas pertanyaannya", "terima kasih telah bertanya di alodokter.com",
    "hai", "halo", "hai ayu atidhira", "baik pria dan wanita masing-masing dapat memiliki permasalahan pada organ reproduksi yang dapat memengaruhi fertilitas atau kesuburan mereka", "halo pandu", "hai syukroni ahmad", "dear halim", "terima kasih dyah islamika atas pertanyaannya", "selamat malam adam achmad hamdallah", "selamat sore", "halo adhe", "halo khoirul", "dear dilly", "halo", "hai frengky", "halo agus", "hai usman", "hai taqqiyah arini", "hai rin", "halo rizki", "halo amirul", "halo nev", "halo anca", "jung hye mun", "mayang sari", "pak yansen", "hi iya"
    "halo aldrian", "halo fikry", "hai zahirah", "hai robby", "halo rinaldi", "hai", "halo ayundya", "hai maulana", "salam", "terima kasih awi nadnad atas pertanyaannya", "halo edo", "halo arhy", "halo mario", "halo deritamu", "halo zainatun", "selamat malam", "halo", "halo lam", "halo icha", "halo", "halo rendi", "halo fitranta", "adam achmad hamdallah", "halo deritamu", "halo lam", "Abie KanGen'Lagi",

    "dr. riza marlina", "dr. nadia nurotul fuadah", "dr. tirtawati wijaya, se", "dr. amadeo d. basfiansa", "dr. riska larasati", "dr. devika y", "dr. tabita p s", "dr. iranita dyantika", "dr. fajar dwi cahyo", "dr. singgih e prasetyo", "dr. sussy listiarsasih", "dr. maria", "dr. novalia arisandy", "dr. deasy larasandi", "dr. iriyanti maya sari barutu", "dr. sarah rizqia", "dr. danny", "dr. setiawan winarso",
    "dr. kresnawati setiono", "dr. yusi capriyanti", "dr. shirly widjaja", "dr. winda indriati", "dr. aditya prayoga", "dr. delvira parinding", "dr. irvandi", "dr. thoriqotil haqqul mauludiyah", "dr. rizki amy lavita", "dr. fenita antonius", "dr. muhammad fadhil", "dr. aldy valentino maehcarenda", "dr. nugraha arief", "dr. sonia loviarny", "dr. nurmarwiyah", "dr. wahyu febrianto", "dr. christian haryanto junaedi",
    "dr. adriana virani jeumpa", "dr. alyssa diandra", "dr. radius kusuma", "dr. celleen rei setiawan", "dr. jati satriyo", "dr. taneya putri zahra", "dr. muliani sukiman", "dr. yuniar cahyania intani", "dr. sylvia djohan", "dr. adhi pasha", "dr. theresia yoshiana", "dr. natasha alexander", "dr. ellysabet dian", "dr. prasetyo", "dr. anthony maleachi", "dr. paramita, m.biomed", "dr. caecilia haryu aryapti",
    "dr. atikah dafri", "dr. tantya marlien", "dr. yoni cahyati", "dr. annisa auli adjani", "dr. lili dwiyani", "dr. rony wijaya", "dr. annes waren", "dr. andika surya atmadja", "dr. luh putu prevyanti", "dr. yosephine. s.", "dr. siti rahmayanti", "dr. deslia anggarini supriyadi", "dr. tri permatadewi", "dr. mesha syafitra", "dr. mira iskandar", "dr. otniel budi krisetya", "dr. nugraha mauluddin", "dr. debby phanggestu",
    "dr. rico n", "dr. agatha dinar", "dr. rievia bahasoean", "dr. tessi ananditya", "dr. nisia putri rinayu", "dr. eunike kiki m. sitompul", "dr. f. sutandi", "dr. andisty ate", "dr. aldo ferly", "dr. aditya pratama", "dr. nofrina", "dr. agnes maureen", "dr. dian paramitasari", "dr. lina saleh", "dr. henry andrean", "dr. jessica winoto", "dr. nadia nurotul fuadah", "dr. tri", "dr. liyadi", "dr. debby phanggestu", "dr. yusi", "dr. dian paramitasari",
    "Dian Paramitasari", "dr. agatha", "dr. yan william", "dr. nofrina", "dr. yusi", "dr. jessica winoto", "dr. agnes maureen", "dian paramitasari dr", "ke ter",
    "dr. yan william", "dr. radius kusuma", "dr. rievia", "dr. miranti iskandar", "dr. debby phanggestu", "dr. yosephine", "dr. aditya pratama", "dr. jessica", "dr. rony wijaya", "dr. yusi", "dr. jessica winoto", "dr. muliani", "dr. yusi", "dr. agatha", "dr. yosephine", "dr. radius kusuma", "dr. muliani sukiman", "dr. dennis jacobus", "dr. debby phanggestu", "dr. miranti iskandar", "dr. rievia", "dr. anandia salsabila", "dr. yan william",
    "dr. otniel budi krisetya", "dr. andika surya", "dr. yusi", "dr. aldo", "dr. nugraha mauluddin", "dr. yosephine", "dr. aditya pratama", "dr. jati satriyo", "dr. debby phanggestu", "dr. tantya marlien", "dr. muliani sukiman", "dr. anandia salsabila", "dr. yan william", "dr. otniel budi krisetya", "dr. andika surya", "dr. yusi", "dr. aldo", "dr. nugraha mauluddin", "dr. yosephine", "dr. aditya pratama", "dr. jati satriyo", "dr. debby phanggestu",
    "dr. tantya marlien", "dr. muliani sukiman", "dr. theresia", "dr. tri", "dr. arnold", "dr. adhi pasha", "dr. nadia nurotul fuadah", "dr. abi", "dr. yuniar", "dr. jessica", "dr. daniel", "dr. lili", "dr. devika yuldharia", "dr. mira", "dr. ulfi", "dr. mesha", "dr. radius kusuma", "dr. deslia", "dr. jessica", "dr. ulfi", "dr. theresia", "dr. lili", "dr. annes", "dr. aldo", "dr. nisia", "dr. adhi p.", "dr. devika y", "dr. yosephine", "dr. jati", "dr. yan william",
    "dr. arnold", "dr. danny", "dr. muhammad fadhil rahmadiansyah", "dr. andika surya", "dr. lili", "dr. yosephine", "dr. rony wijaya", "dr. theresia", "dr. adhi p.", "dr. alyssa", "dr. adriana", "dr. sonia l", "dr. devika yuldharia", "dr. jati", "dr. yoni cahyati", "dr. siti rahmayanti", "dr. andisty ate", "dr. henry andrean", "dr. anandia salsabila", "dr. alyssa", "dr. natasha", "dr. prasetyo", "dr. yan william", "dr. radhianie djan", "dr. jati", "dr. yusi", "dr. ami",
    "dr. nadia nurotul fuadah", "dr. caecilia haryu aryapti", "dr. danny", "dr. tantya", "dr. yoni cahyati", "dr. anissa", "dr. ulfi", "dr. n. k. arief", "dr. lia n. amalina", "dr. rony wijaya", "dr. yosephine", "dr. siti rahmayanti", "dr. adhi p.", "dr. luh putu previyanti dharmaputri", "dr. andika surya", "dr. anthony", "dr. n. k. arief", "dr. nadia nurotul fuadah", "dr. adhi p.", "dr. alyssa", "dr. celleen rei setiawan", "dr. christian haryanto", "dr. irvandi", "dr. jati",
    "dr. theresia", "dr. natasha", "dr. devika y", "dr. sonia l", "dr. aloisia", "dr. dian", "dr. sylvia", "dr. prasetyo", "dr. anthony", "dr. lia n. amalina", "dr. asri", "dr. radhianie djan", "dr. aldy valentino", "dr. irna cecilia", "dr. anandia salsabila", "dr. devika y", "dr. iriyanti", "dr. aldy valentino", "dr. n. k. arieff", "dr. sonia l", "dr. amadeo d. basfiansa", "dr. irna cecilia", "dr. nadia nurotul fuadah", "dr. danny", "dr. nurmarwiyah", "dr. christian haryanto",
    "dr. aloisia", "dr. budiono", "dr. alyssa", "dr. ulfi", "dr. radius kusuma", "dr. celleen", "dr. taneya putri zahra", "dr. muliani sukiman", "dr. yuniar", "dr. sylvia", "dr. jati", "dr. farah", "dr. amadeo d. basfiansa", "dr. mega", "dr. denisa", "dr. kresnawati wahyu setiono", "dr. shirly", "dr. winda indriati", "dr. thoriqotil h. m.", "dr. irvandi", "dr. delvira", "dr. amy", "dr. muhammad fadhil", "dr. anandia salsabila", "dr. bla bla", "dr. hangpi", "dr. menteri", "dr. bagas",
    "dr. stanley", "dr. rina", "dr. juan", "dr. abdul", "dr. fifie gabriel", "dr. rejeki", "dr. miya", "dr. rangga", "dr. stefanus reinaldi", "dr. ibnuwepe", "dr. yanuar97", "dr. gen", "dr. jhoni", "dr. veri", "dr. budiono", "dr. rio", "dr. sussy", "dr. nova", "dr. tirta wijaya", "dr. iranita", "dr. ainul"

}

lemmatizer = WordNetLemmatizer()
# factor = StemmerFactory()
# stemmer = factor.create_stemmer()

# function preprocess
def preprocess(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    
    for phrase in custom_stopwords_phrases:
        text = text.replace(phrase.lower(), '')

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    # tokens = [stemmer.stem(word) for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in custom_stopwords]
    return " ".join(tokens)

# read dataset
df = pd.read_csv('static/assets/cleaned_data_qna.csv', encoding='utf-8')
df.dropna(subset=['Member Topic Content','Doctor Content'], inplace=True)

# Preprocess question and answer
df['processed_member_topic_content'] = df['Member Topic Content'].apply(preprocess)
df['processed_doctor_content'] = df['Doctor Content'].apply(preprocess)

# TF-IDF dan vektor
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['processed_member_topic_content'])

def get_answer(user_input, threshold=0.6):
    preprocessed_input = preprocess(user_input)
    user_vector = vectorizer.transform([preprocessed_input])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    print(f"[DEBUG] Best score: {best_score:.4f} | Index: {best_match_index}")

    if best_score < threshold:

        response = random.choice([
            "Maaf, saya kurang mengerti. Bisakah kamu jelaskan lebih detail?",
            "Pertanyaanmu belum bisa saya jawab. Coba tanyakan hal lain ya!",
            "Maaf, saya tidak memiliki informasi yang cukup relevan untuk pertanyaan tersebut."
        ])
        return response
    
    answer = df.iloc[best_match_index]['processed_doctor_content']

# explainable chat
    matched_question = df.iloc[best_match_index]['processed_member_topic_content']
    feature_name = vectorizer.get_feature_names_out()
    non_zero_weight = user_vector.toarray()[0]
    keyword_used = {
        feature_name[i]: round(non_zero_weight[i], 4)
        for i in non_zero_weight.nonzero()[0]
    }
    print(f"Input Keywords : {keyword_used}\n")
    print(f"mached_question: {matched_question}\n")
    print(f"similarity score: {best_score:.4f}\n")
    print(f"answer: {answer}\n")

    return answer