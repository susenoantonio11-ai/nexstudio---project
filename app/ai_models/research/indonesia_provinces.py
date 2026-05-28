"""
Indonesia Province Registry
===========================
Province codes used as the spatial unit of analysis. Following Permendagri
No. 137/2017 (BPS-aligned) plus the four DOB (Daerah Otonom Baru) created
by UU 14/2022, UU 15/2022, UU 16/2022 (Papua expansion):

  Papua Selatan (Merauke), Papua Tengah (Nabire), Papua Pegunungan (Jayawijaya),
  Papua Barat Daya (Sorong)

Total = 38 provinces. Each entry carries:
  id           : ISO-style code (idn-jw, idn-jb, ...)
  bps_code     : 2-digit BPS province code (used by data.bnpb.go.id CSV)
  name_id      : Indonesian name (BPS canonical)
  name_en      : English name (for paper writing)
  bbox         : [west, south, east, north] in EPSG:4326 — used for GEE clip
  iso_3166_2   : ISO 3166-2 sub-national code

Bounding boxes are nominal — adequate for clipping CHIRPS / MODIS but for
final paper figures replace with the official BPS shapefile centroids.
"""
from typing import Dict, List, Any

INDONESIA_PROVINCES: List[Dict[str, Any]] = [
    {"id": "idn-ac", "bps_code": "11", "name_id": "Aceh",                "name_en": "Aceh",                  "iso_3166_2": "ID-AC", "bbox": [95.0,  2.0, 98.5,  6.1]},
    {"id": "idn-su", "bps_code": "12", "name_id": "Sumatera Utara",      "name_en": "North Sumatra",         "iso_3166_2": "ID-SU", "bbox": [97.0,  1.3,100.5,  4.3]},
    {"id": "idn-sb", "bps_code": "13", "name_id": "Sumatera Barat",      "name_en": "West Sumatra",          "iso_3166_2": "ID-SB", "bbox": [98.6, -3.4,101.9,  0.9]},
    {"id": "idn-ri", "bps_code": "14", "name_id": "Riau",                "name_en": "Riau",                  "iso_3166_2": "ID-RI", "bbox": [100.2,-1.0,103.6,  2.5]},
    {"id": "idn-ja", "bps_code": "15", "name_id": "Jambi",               "name_en": "Jambi",                 "iso_3166_2": "ID-JA", "bbox": [101.0,-2.8,104.5, -0.7]},
    {"id": "idn-sl", "bps_code": "16", "name_id": "Sumatera Selatan",    "name_en": "South Sumatra",         "iso_3166_2": "ID-SS", "bbox": [102.2,-4.9,106.0, -1.2]},
    {"id": "idn-be", "bps_code": "17", "name_id": "Bengkulu",            "name_en": "Bengkulu",              "iso_3166_2": "ID-BE", "bbox": [101.0,-5.4,104.3, -2.2]},
    {"id": "idn-la", "bps_code": "18", "name_id": "Lampung",             "name_en": "Lampung",               "iso_3166_2": "ID-LA", "bbox": [103.4,-6.0,106.0, -3.7]},
    {"id": "idn-bb", "bps_code": "19", "name_id": "Bangka Belitung",     "name_en": "Bangka Belitung",       "iso_3166_2": "ID-BB", "bbox": [105.0,-3.4,108.5, -1.5]},
    {"id": "idn-kr", "bps_code": "21", "name_id": "Kepulauan Riau",      "name_en": "Riau Islands",          "iso_3166_2": "ID-KR", "bbox": [103.0,-1.0,109.0,  4.5]},
    {"id": "idn-jk", "bps_code": "31", "name_id": "DKI Jakarta",         "name_en": "Jakarta",               "iso_3166_2": "ID-JK", "bbox": [106.6,-6.4,107.0, -6.1]},
    {"id": "idn-jb", "bps_code": "32", "name_id": "Jawa Barat",          "name_en": "West Java",             "iso_3166_2": "ID-JB", "bbox": [105.9,-7.8,108.9, -5.9]},
    {"id": "idn-jt", "bps_code": "33", "name_id": "Jawa Tengah",         "name_en": "Central Java",          "iso_3166_2": "ID-JT", "bbox": [108.6,-8.4,111.7, -5.7]},
    {"id": "idn-yo", "bps_code": "34", "name_id": "DI Yogyakarta",       "name_en": "Yogyakarta",            "iso_3166_2": "ID-YO", "bbox": [110.0,-8.3,110.9, -7.6]},
    {"id": "idn-ji", "bps_code": "35", "name_id": "Jawa Timur",          "name_en": "East Java",             "iso_3166_2": "ID-JI", "bbox": [110.9,-8.8,114.7, -6.7]},
    {"id": "idn-bt", "bps_code": "36", "name_id": "Banten",              "name_en": "Banten",                "iso_3166_2": "ID-BT", "bbox": [105.1,-7.0,106.8, -5.8]},
    {"id": "idn-ba", "bps_code": "51", "name_id": "Bali",                "name_en": "Bali",                  "iso_3166_2": "ID-BA", "bbox": [114.4,-8.9,115.7, -8.0]},
    {"id": "idn-nb", "bps_code": "52", "name_id": "Nusa Tenggara Barat", "name_en": "West Nusa Tenggara",    "iso_3166_2": "ID-NB", "bbox": [115.7,-9.2,119.4, -8.0]},
    {"id": "idn-nt", "bps_code": "53", "name_id": "Nusa Tenggara Timur", "name_en": "East Nusa Tenggara",    "iso_3166_2": "ID-NT", "bbox": [118.9,-11.0,125.2,-8.0]},
    {"id": "idn-kb", "bps_code": "61", "name_id": "Kalimantan Barat",    "name_en": "West Kalimantan",       "iso_3166_2": "ID-KB", "bbox": [108.6,-3.0,114.2,  2.1]},
    {"id": "idn-kt", "bps_code": "62", "name_id": "Kalimantan Tengah",   "name_en": "Central Kalimantan",    "iso_3166_2": "ID-KT", "bbox": [110.7,-3.6,115.7,  0.4]},
    {"id": "idn-ks", "bps_code": "63", "name_id": "Kalimantan Selatan",  "name_en": "South Kalimantan",      "iso_3166_2": "ID-KS", "bbox": [114.4,-4.2,117.0, -1.2]},
    {"id": "idn-ki", "bps_code": "64", "name_id": "Kalimantan Timur",    "name_en": "East Kalimantan",       "iso_3166_2": "ID-KI", "bbox": [115.0,-2.7,119.0,  2.6]},
    {"id": "idn-ku", "bps_code": "65", "name_id": "Kalimantan Utara",    "name_en": "North Kalimantan",      "iso_3166_2": "ID-KU", "bbox": [115.5, 2.6,118.5,  4.7]},
    {"id": "idn-su2","bps_code": "71", "name_id": "Sulawesi Utara",      "name_en": "North Sulawesi",        "iso_3166_2": "ID-SA", "bbox": [123.0, 0.3,127.0,  4.8]},
    {"id": "idn-st", "bps_code": "72", "name_id": "Sulawesi Tengah",     "name_en": "Central Sulawesi",      "iso_3166_2": "ID-ST", "bbox": [119.4,-3.0,124.5,  1.5]},
    {"id": "idn-ss2","bps_code": "73", "name_id": "Sulawesi Selatan",    "name_en": "South Sulawesi",        "iso_3166_2": "ID-SN", "bbox": [118.7,-7.8,121.7, -1.5]},
    {"id": "idn-sg", "bps_code": "74", "name_id": "Sulawesi Tenggara",   "name_en": "Southeast Sulawesi",    "iso_3166_2": "ID-SG", "bbox": [120.4,-6.2,124.5, -2.6]},
    {"id": "idn-go", "bps_code": "75", "name_id": "Gorontalo",           "name_en": "Gorontalo",             "iso_3166_2": "ID-GO", "bbox": [121.4, 0.4,123.4,  1.1]},
    {"id": "idn-sr", "bps_code": "76", "name_id": "Sulawesi Barat",      "name_en": "West Sulawesi",         "iso_3166_2": "ID-SR", "bbox": [118.6,-3.5,120.0, -1.0]},
    {"id": "idn-ma", "bps_code": "81", "name_id": "Maluku",              "name_en": "Maluku",                "iso_3166_2": "ID-MA", "bbox": [124.0,-8.6,135.0, -2.4]},
    {"id": "idn-mu", "bps_code": "82", "name_id": "Maluku Utara",        "name_en": "North Maluku",          "iso_3166_2": "ID-MU", "bbox": [124.0,-2.5,129.7,  2.7]},
    {"id": "idn-pa", "bps_code": "91", "name_id": "Papua",               "name_en": "Papua",                 "iso_3166_2": "ID-PA", "bbox": [134.0,-4.2,141.0, -2.0]},
    {"id": "idn-pb", "bps_code": "92", "name_id": "Papua Barat",         "name_en": "West Papua",            "iso_3166_2": "ID-PB", "bbox": [131.0,-4.0,135.0,  0.5]},
    # 2022 DOB (Papua expansion)
    {"id": "idn-ps", "bps_code": "93", "name_id": "Papua Selatan",       "name_en": "South Papua",           "iso_3166_2": "ID-PS", "bbox": [137.0,-9.0,141.0, -5.5]},
    {"id": "idn-pt", "bps_code": "94", "name_id": "Papua Tengah",        "name_en": "Central Papua",         "iso_3166_2": "ID-PT", "bbox": [134.0,-4.5,138.5, -2.5]},
    {"id": "idn-pg", "bps_code": "95", "name_id": "Papua Pegunungan",    "name_en": "Highland Papua",        "iso_3166_2": "ID-PG", "bbox": [137.0,-5.5,141.0, -3.0]},
    {"id": "idn-pd", "bps_code": "96", "name_id": "Papua Barat Daya",    "name_en": "Southwest Papua",       "iso_3166_2": "ID-PD", "bbox": [129.0,-2.0,134.0,  0.5]},
]


def list_province_ids() -> List[str]:
    return [p["id"] for p in INDONESIA_PROVINCES]


def get_province(province_id: str) -> Dict[str, Any]:
    for p in INDONESIA_PROVINCES:
        if p["id"] == province_id or p["bps_code"] == province_id:
            return p
    raise KeyError(f"Province not found: {province_id}")


PROVINCE_COUNT = len(INDONESIA_PROVINCES)
