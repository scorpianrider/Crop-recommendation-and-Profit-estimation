# config/rainfall_normals.py
# Monthly rainfall normals (mm) for Tamil Nadu districts.
# Format: district -> [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]

DISTRICT_RAINFALL_NORMALS = {
    "Thanjavur":        [13,  7,   5,   30,  55,  44,  87,  85,  121, 135, 154, 130],
    "Tiruvarur":        [20,  8,   6,   26,  60,  48,  90,  118, 130, 140, 150, 105],
    "Nagapattinam":     [30,  12,  8,   35,  70,  55,  95,  130, 150, 160, 165, 130],
    "Tiruchirappalli":  [15,  7,   6,   25,  55,  34,  66,  80,  103, 115, 123, 64],
    "Cuddalore":        [25,  10,  7,   30,  60,  50,  85,  120, 135, 140, 165, 115],
    "Viluppuram":       [20,  8,   5,   28,  52,  42,  75,  105, 120, 145, 160, 90],
    "Thiruvallur":      [18,  8,   5,   25,  45,  38,  65,  95,  110, 135, 160, 85],
    "Erode":            [6,   4,   8,   35,  70,  40,  50,  65,  85,  110, 55,  15],
    "Coimbatore":       [8,   5,   10,  40,  80,  45,  55,  70,  90,  120, 65,  20],
    "Salem":            [10,  6,   9,   38,  72,  38,  52,  68,  88,  115, 60,  18],
    "Namakkal":         [8,   5,   8,   36,  68,  36,  50,  65,  85,  110, 55,  16],
    "Dindigul":         [12,  6,   10,  42,  78,  42,  55,  72,  92,  120, 68,  22],
    "Karur":            [9,   5,   8,   35,  68,  36,  50,  65,  85,  108, 55,  16],
    "Madurai":          [12,  6,   9,   40,  72,  38,  52,  68,  88,  118, 62,  20],
    "Virudhunagar":     [14,  7,   10,  42,  75,  40,  55,  70,  90,  122, 65,  22],
    "Sivagangai":       [14,  7,   9,   40,  70,  38,  52,  68,  88,  118, 62,  20],
    "Ramanathapuram":   [20,  8,   8,   38,  62,  32,  45,  60,  80,  130, 115, 40],
    "Theni":            [10,  6,   12,  50,  90,  55,  65,  82,  100, 130, 70,  24],
    "Tirunelveli":      [18,  8,   10,  40,  80,  45,  55,  70,  90,  130, 90,  35],
    "Tiruppur":         [8,   5,   9,   38,  72,  40,  52,  67,  87,  112, 58,  18],
    "Kanyakumari":      [48,  22,  40,  85,  120, 142, 150, 160, 160, 167, 125, 55],
    "Krishnagiri":      [8,   5,   7,   32,  65,  35,  48,  62,  82,  108, 52,  14],
    "Dharmapuri":       [7,   4,   7,   30,  62,  33,  46,  60,  80,  105, 50,  13],
    "Nilgiris":         [40,  30,  55,  110, 160, 160, 180, 180, 185, 187, 180, 75],
}


def get_rainfall_normal(district: str, month: int) -> float:
    TN_STATE_AVG = [15, 7, 7, 35, 68, 42, 62, 85, 100, 155, 140, 55]
    normals = DISTRICT_RAINFALL_NORMALS.get(district, TN_STATE_AVG)
    return float(normals[month - 1])
