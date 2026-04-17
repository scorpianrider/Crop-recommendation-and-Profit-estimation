# utils/soil_npk_lookup.py

SOIL_NPK_LOOKUP = {
    ("Red",      "Sandy"):  {"N": 13, "P": 10, "K": 4},
    ("Red",      "Loamy"):  {"N": 23, "P": 23, "K": 10},
    ("Red",      "Clayey"): {"N": 25, "P": 13, "K": 23},
    ("Black",    "Sandy"):  {"N": 25, "P": 36, "K": 4},
    ("Black",    "Loamy"):  {"N": 37, "P": 25, "K": 23},
    ("Black",    "Clayey"): {"N": 37, "P": 36, "K": 23},
    ("Brown",    "Sandy"):  {"N": 39, "P": 10, "K": 4},
    ("Brown",    "Loamy"):  {"N": 25, "P": 23, "K": 5},
    ("Brown",    "Clayey"): {"N": 25, "P": 10, "K": 35},
    ("Alluvial", "Sandy"):  {"N": 25, "P": 23, "K": 23},
    ("Alluvial", "Loamy"):  {"N": 37, "P": 36, "K": 35},
    ("Alluvial", "Clayey"): {"N": 37, "P": 36, "K": 23},
    # Map "Dark" (used in the frontend) to Black as closest match
    ("Dark",     "Sandy"):  {"N": 25, "P": 36, "K": 4},
    ("Dark",     "Loamy"):  {"N": 37, "P": 25, "K": 23},
    ("Dark",     "Clayey"): {"N": 37, "P": 36, "K": 23},
    # Map "Clay" (frontend label) same as "Clayey"
    ("Red",      "Clay"):   {"N": 25, "P": 13, "K": 23},
    ("Black",    "Clay"):   {"N": 37, "P": 36, "K": 23},
    ("Brown",    "Clay"):   {"N": 25, "P": 10, "K": 35},
    ("Dark",     "Clay"):   {"N": 37, "P": 36, "K": 23},
}

VALID_SOIL_COLOURS  = ["Red", "Black", "Brown", "Alluvial", "Dark"]
VALID_SOIL_TEXTURES = ["Sandy", "Loamy", "Clayey", "Clay"]


def estimate_npk(soil_colour: str, soil_texture: str) -> dict:
    colour  = soil_colour.strip().capitalize()
    texture = soil_texture.strip().capitalize()

    # Normalise "Clay" → "Clayey" if needed
    if texture == "Clay":
        texture = "Clayey"

    key = (colour, texture)
    if key not in SOIL_NPK_LOOKUP:
        # Fallback: return mid-range values
        return {"N": 25, "P": 20, "K": 15}
    return SOIL_NPK_LOOKUP[key].copy()
