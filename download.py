import pandas as pd
import pyarrow.parquet as pq
import requests
import io
from datetime import date

# =========================
# AYARLAR
# =========================

DATA_TYPE = "mobile"   # "mobile" veya "fixed"

# Türkiye bounding box
TURKEY_BBOX = {
    "lat_min": 35.5,
    "lat_max": 42.5,
    "lon_min": 25.5,
    "lon_max": 45.0
}

# tile_x (lon) ve tile_y (lat) sütunları Q3 2023'ten beri mevcut
# avg_lat_down_ms / avg_lat_up_ms Q4 2022'den beri mevcut (sparse)
COLUMNS = [
    "avg_d_kbps",
    "avg_u_kbps",
    "avg_lat_ms",
    "avg_lat_down_ms",
    "avg_lat_up_ms",
    "tests",
    "devices",
    "tile_x",
    "tile_y",
]

BASE_URL = "https://ookla-open-data.s3.amazonaws.com/parquet/performance"

QUARTER_MONTHS = {1: 1, 2: 4, 3: 7, 4: 10}


# =========================
# URL OLUŞTURMA
# =========================

def build_url(year, quarter):
    month = QUARTER_MONTHS[quarter]
    date_str = f"{year}-{month:02d}-01"
    filename = f"{date_str}_performance_{DATA_TYPE}_tiles.parquet"
    return f"{BASE_URL}/type={DATA_TYPE}/year={year}/quarter={quarter}/{filename}"


def find_latest_quarter():
    """Mevcut en son çeyreği bulur (HEAD isteğiyle kontrol eder)."""
    today = date.today()
    year = today.year
    current_q = (today.month - 1) // 3  # tamamlanmış çeyrek

    if current_q == 0:
        year -= 1
        current_q = 4

    for _ in range(8):
        url = build_url(year, current_q)
        print(f"🔍 Kontrol ediliyor: {year} Q{current_q} ...")
        try:
            resp = requests.head(url, timeout=10)
            if resp.status_code == 200:
                size_mb = int(resp.headers.get("Content-Length", 0)) / 1024 / 1024
                print(f"✅ Mevcut: {year} Q{current_q}  ({size_mb:.0f} MB)")
                return year, current_q, url
        except Exception:
            pass

        current_q -= 1
        if current_q == 0:
            current_q = 4
            year -= 1

    raise Exception("Son 8 çeyrekte mevcut veri bulunamadı!")


# =========================
# VERİ İNDİRME (Streaming + Pyarrow Filter)
# =========================

def load_latest_quarter():
    year, quarter, url = find_latest_quarter()
    print(f"\n📥 {year} Q{quarter} indiriliyor (sadece Türkiye bbox filtresi uygulanıyor)...")
    print(f"   URL: {url}\n")

    # Dosyayı bellekte indir
    print("   Dosya indiriliyor...", end="", flush=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunks = []
    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunk
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r   İndiriliyor: {downloaded/1024/1024:.1f} / {total/1024/1024:.1f} MB ({pct:.0f}%)    ", end="", flush=True)

    print()
    buf = io.BytesIO(b"".join(chunks))

    # Pyarrow filter ile Türkiye bbox uygula
    filters = [
        ("tile_x", ">=", TURKEY_BBOX["lon_min"]),
        ("tile_x", "<=", TURKEY_BBOX["lon_max"]),
        ("tile_y", ">=", TURKEY_BBOX["lat_min"]),
        ("tile_y", "<=", TURKEY_BBOX["lat_max"]),
    ]

    table = pq.read_table(buf, columns=COLUMNS, filters=filters)
    df = table.to_pandas()
    print(f"✅ Yüklendi: {len(df):,} satır (Türkiye bbox filtresi uygulandı)")
    return df, year, quarter


# =========================
# TÜRKİYE FİLTRELEME
# =========================

def filter_turkey(df):
    """Pyarrow filter yaklaşık olduğundan bbox'ı kesin uygula."""
    print("🇹🇷 Türkiye bbox kesin filtre uygulanıyor...")

    df_tr = df[
        (df["tile_y"] >= TURKEY_BBOX["lat_min"]) &
        (df["tile_y"] <= TURKEY_BBOX["lat_max"]) &
        (df["tile_x"] >= TURKEY_BBOX["lon_min"]) &
        (df["tile_x"] <= TURKEY_BBOX["lon_max"])
    ]

    print(f"✅ Türkiye veri sayısı: {len(df_tr):,}")
    return df_tr.reset_index(drop=True)


# =========================
# ANALİZ
# =========================

def analyze(df):
    print("\n📊 Özet istatistik:")

    stats = df[["avg_d_kbps", "avg_u_kbps", "avg_lat_ms"]].describe()
    print(stats)


# =========================
# MAIN
# =========================

def main():
    print("🚀 Ookla en son veri indirme başlıyor...\n")

    df, year, quarter = load_latest_quarter()

    print(f"\n🌐 Toplam veri: {len(df):,}")

    df_tr = filter_turkey(df)

    analyze(df_tr)

    output_file = f"ookla_turkey_{year}_Q{quarter}.parquet"
    print("\n💾 Kaydediliyor...")

    df_tr.to_parquet(output_file, engine="pyarrow")

    print(f"✅ Tamamlandı: {output_file}")


if __name__ == "__main__":
    main()
