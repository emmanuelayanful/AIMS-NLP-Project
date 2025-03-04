african_language_map = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Fulah": "ff",
    "Hausa": "ha",
    "Igbo": "ig",
    "Lingala": "ln",
    "Malagasy": "mg",
    "Northern Sotho": "ns",
    "Somali": "so",
    "Swati": "ss",
    "Swahili": "sw",
    "Xhosa": "xh",
    "Yoruba": "yo",
    "Zulu": "zu",
    "Ganda": "lg",
    "Wolof": "wo",
    "Shona": "sn",
    "Tswana": "tn",
    "Ewe": "ee",
    "Bambara": "bm",
    "Kinyarwanda": "rw",
    "Twi": "tw",
    "Oromo": "om",
    "Tigrinya": "ti",
    "Sesotho": "st",
    "English": "en"
}

# Function to get the language code from language name
def get_language_code(language_name):
    return african_language_map.get(language_name, None)  # Returns None if the language is not found
