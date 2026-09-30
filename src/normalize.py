import re
import unicodedata


LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "plc", "pvt", "private",
    "pte", "gmbh", "sarl", "sas", "sa", "ag",
}

ADDRESS_MAP = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "parkway": "pkwy",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
}

COUNTRY_MAP = {
    "usa": "us",
    "unitedstates": "us",
    "unitedstatesofamerica": "us",
    "u s a": "us",
    "india": "in",
    "ind": "in",
    "france": "fr",
    "fra": "fr",
}


def ascii_fold(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_text(value: str) -> str:
    text = ascii_fold(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: str) -> str:
    tokens = normalize_text(value).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_address(value: str) -> str:
    return " ".join(
        ADDRESS_MAP.get(token, token)
        for token in normalize_text(value).split()
    )


def normalize_country(value: str) -> str:
    text = normalize_text(value)
    compact = text.replace(" ", "")
    return COUNTRY_MAP.get(text, COUNTRY_MAP.get(compact, compact))


def postal_code(value: str) -> str:
    text = ascii_fold(value)
    matches = re.findall(r"\b\d{5,6}\b", text)
    return matches[0] if matches else ""


def enrich(df):
    result = df.copy()
    result["norm_name"] = result["business_name"].map(normalize_name)
    result["norm_address"] = result["business_address"].map(normalize_address)
    result["norm_country"] = result["country"].map(normalize_country)
    result["postal"] = result["business_address"].map(postal_code)
    result["name_tokens"] = result["norm_name"].map(lambda x: tuple(x.split()))
    result["address_tokens"] = result["norm_address"].map(lambda x: tuple(x.split()))
    result["name_prefix"] = result["norm_name"].map(lambda x: x[:4] if x else "")
    return result
