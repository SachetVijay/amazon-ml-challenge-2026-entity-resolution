"""Record normalisation: script-agnostic, country-agnostic (France is unseen in train).

Every record gets several *views* so downstream similarity is robust to the
generator's noise (typos, transliteration, abbreviation swaps, reordering, junk):

  name_clean   ascii, lowercase, punctuation-free, noise words removed, abbreviations expanded
  name_core    name_clean minus legal-form / generic business words ("private limited", "llc", "services")
  name_skel    consonant skeleton per core token  (praivet ~ private -> prvt, phuds ~ foods -> fts)
  legal        canonical legal-form set (llc, inc, pvt_ltd, sarl, ...)
  addr_clean   ascii, lowercase, abbreviations expanded, junk ("null", "##", "n/a") removed
  addr_nums    set of number tokens (house / plot / flat numbers)
  addr_skel    skeleton tokens of the address
"""
from __future__ import annotations

import re
import unicodedata

from anyascii import anyascii

# --- legal forms (value = canonical group). Includes French forms: test has France.
LEGAL = {
    "llc": "llc", "l.l.c": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "limitet": "ltd", "lp": "lp", "llp": "llp",
    "pllc": "pllc", "pc": "pc", "plc": "plc", "pvt": "pvt", "private": "pvt", "praivet": "pvt", "pte": "pvt",
    "gmbh": "gmbh", "sa": "sa", "sas": "sas", "sasu": "sas", "sarl": "sarl", "eurl": "sarl", "sci": "sci",
    "snc": "snc", "scop": "scop", "opc": "opc",
}
# generic words the generator sprinkles in / that carry little identity signal
GENERIC = {
    "the", "and", "of", "services", "service", "center", "centre", "partners", "group", "enterprises",
    "enterprise", "solutions", "holdings", "international", "global", "company", "com", "www", "net", "org",
    "in", "co", "sri", "shri", "shree", "smt", "m", "s", "ms", "mr", "mrs", "dr", "dba", "aka", "formerly",
    "nee", "doing", "business", "as", "t", "a", "et", "de", "du", "des", "la", "le", "les",
    # French equivalents of the generator's filler words (the test set adds France)
    "fils", "freres", "cie", "compagnie", "groupe", "developpement", "societe", "ste", "ets",
    "etablissements", "associes", "partenaires", "international", "internationale", "centre", "l", "d",
}
NAME_ABBR = {"&": "and", "intl": "international", "svcs": "services", "svc": "service", "ctr": "center",
             "mfg": "manufacturing", "assoc": "associates", "bros": "brothers", "natl": "national",
             "tech": "technologies", "univ": "university", "hosp": "hospital", "mgmt": "management"}

ADDR_ABBR = {
    # US
    "st": "st", "street": "st", "saint": "st", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "ln": "lane", "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "trl": "trail", "ter": "terrace", "sq": "square", "mt": "mount", "ft": "fort", "apt": "apartment",
    "ste": "suite", "fl": "floor", "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "twp": "township",
    # India
    "appt": "apartment", "apts": "apartments", "nr": "near", "opp": "opposite", "ngr": "nagar", "mg": "mg",
    "no": "number", "hno": "house number", "h": "house", "c/o": "care of", "dist": "district", "tal": "taluka",
    # France
    "bd": "boulevard", "bld": "boulevard", "r": "rue", "chem": "chemin", "rte": "route", "imp": "impasse",
    "fbg": "faubourg", "all": "allee", "ch": "chemin", "q": "quai", "crs": "cours", "sainte": "ste",
    "av": "avenue", "pass": "passage", "res": "residence", "lot": "lotissement", "zi": "zone industrielle",
    "za": "zone artisanale",
}
ADDR_JUNK = {"null", "na", "n/a", "none", "po", "box", "pmb", "unit", "suite", "apartment", "number", "door",
             "flat", "floor", "house", "care", "of", "near", "opposite", "cdp", "city", "township", "town",
             "village", "district", "taluka", "de", "la", "le", "les", "du", "des", "d", "l", "bis", "ter"}

# Canonical state names (US + India). Built-in general knowledge, not an external lookup.
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "mh": "maharashtra", "dl": "delhi", "ka": "karnataka", "up": "uttar pradesh", "gj": "gujarat",
    "wb": "west bengal", "tg": "telangana", "ts": "telangana", "tn": "tamil nadu", "hr": "haryana",
    "kl": "kerala", "ap": "andhra pradesh", "rj": "rajasthan", "mp": "madhya pradesh", "pb": "punjab",
    "br": "bihar", "or": "odisha", "od": "odisha", "jh": "jharkhand", "cg": "chhattisgarh", "as": "assam",
    "uk": "uttarakhand", "hp": "himachal pradesh", "ga": "goa", "jk": "jammu and kashmir", "ch": "chandigarh",
}


# France: 13 metropolitan regions and their departments (static domain knowledge, like US_STATES).
# The data generator swaps region <-> department names; both map to one canonical region token.
FR_REGIONS = {
    "auvergne rhone alpes": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute loire",
                             "puy de dome", "rhone", "savoie", "haute savoie"],
    "bourgogne franche comte": ["cote d or", "cote dor", "doubs", "jura", "nievre", "haute saone", "saone et loire",
                                "yonne", "territoire de belfort"],
    "bretagne": ["cotes d armor", "cotes darmor", "finistere", "ille et vilaine", "morbihan"],
    "centre val de loire": ["cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "corse": ["corse du sud", "haute corse"],
    "grand est": ["ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse", "moselle", "bas rhin",
                  "haut rhin", "vosges"],
    "hauts de france": ["aisne", "nord", "oise", "pas de calais", "somme"],
    "ile de france": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
                      "val de marne", "val d oise", "val doise"],
    "normandie": ["calvados", "eure", "manche", "orne", "seine maritime"],
    "nouvelle aquitaine": ["charente", "charente maritime", "correze", "creuse", "dordogne", "gironde", "landes",
                           "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne", "haute vienne"],
    "occitanie": ["ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot", "lozere",
                  "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "pays de la loire": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "provence alpes cote d azur": ["alpes de haute provence", "hautes alpes", "alpes maritimes", "bouches du rhone",
                                   "var", "vaucluse", "provence alpes cote dazur", "paca"],
}
# phrase -> region code; longest phrases first so "loire atlantique" wins over "loire"
FR_PHRASES = sorted({**{r: r for r in FR_REGIONS}, **{d: r for r, ds in FR_REGIONS.items() for d in ds}}.items(),
                    key=lambda kv: -len(kv[0]))
# single short department names that are also common words are only trusted as a whole address component
FR_AMBIGUOUS = {"ain", "var", "lot", "gard", "cher", "aube", "eure", "orne", "manche", "marne", "somme", "vienne",
                "loire", "indre", "paris", "nord", "jura", "oise", "aude", "gers", "tarn", "meuse", "rhone"}

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9/ ]+")
_DIGITS = re.compile(r"\d+")


# Indian state names written in native scripts (all 16 forms present in the training data)
IN_NATIVE = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "उत्तर प्रदेश": "uttar pradesh", "ಕರ್ನಾಟಕ": "karnataka",
    "தமிழ்நாடு": "tamil nadu", "পশ্চিমবঙ্গ": "west bengal", "ગુજરાત": "gujarat", "తెలంగాణ": "telangana",
    "हरियाणा": "haryana", "राजस्थान": "rajasthan", "കേരളം": "kerala", "मध्य प्रदेश": "madhya pradesh",
    "बिहार": "bihar", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
}
# mojibake from UTF-8 read as Latin-1: "Â" (NBSP), "â" (apostrophe), "Ï¿½" (replacement char)
_MOJIBAKE = re.compile(r"Ï¿½|ï¿½|Â|â(?=[A-Za-z])")
_NUMERO = re.compile(r"\b[Nn]\s*[°º]\s*")          # French/Spanish "N° 12" numbering marker
_DEGREE = re.compile(r"[°º]")


_MOJI_APOS = re.compile(r"(?<=[A-Za-z])[Ââ](?=[A-Za-z])")   # QUEENÂS -> QUEENS (mojibake apostrophe)


def to_ascii(s: str) -> str:
    s = _MOJIBAKE.sub(" ", _MOJI_APOS.sub("", s or ""))
    s = unicodedata.normalize("NFKC", s)
    s = _DEGREE.sub(" ", _NUMERO.sub(" ", s))           # anyascii would turn "°" into "deg" ("N°198" -> "ndeg198")
    return anyascii(s).lower()


def _tokens(s: str) -> list[str]:
    s = s.replace("&", " and ").replace("'", "")
    return _NONALNUM.sub(" ", s).split()


def skeleton(tok: str) -> str:
    """Consonant skeleton robust to transliteration + vowel typos: private/praivet -> prvt."""
    if tok.isdigit():
        return tok
    t = tok.replace("ph", "f").replace("ks", "x").replace("ck", "k").replace("sh", "s").replace("th", "t")
    t = t.translate(str.maketrans("cqzwvdgb", "kksvvtjp"))
    t = re.sub(r"[aeiouyh]", "", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return t or tok[:1]


def norm_name(raw: str) -> dict:
    s = to_ascii(raw)
    s = re.sub(r"https?://|www\.|\.(com|net|org|in|fr|co)\b", " ", s)
    s = re.sub(r"\b(formerly|nee|aka|dba|doing business as|d/b/a)\b", " ", s)
    s = re.sub(r"#\s*\d+", " ", s)                        # store numbers like "#17274"
    s = s.replace("p.c.", "pc").replace("l.l.c.", "llc").replace("l.l.p.", "llp").replace("o.d.", "od")
    toks = [NAME_ABBR.get(t, t) for t in _tokens(s)]
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    clean = [t for t in toks if t not in {"m/s"}]
    core = [t for t in clean if t not in LEGAL and t not in GENERIC] or [t for t in clean if t not in LEGAL] or clean
    return {
        "name_clean": " ".join(clean),
        "name_core": " ".join(core),
        "name_sorted": " ".join(sorted(core)),
        "name_skel": " ".join(skeleton(t) for t in core),
        "legal": " ".join(legal),
    }


def french_regions(s_ascii: str) -> tuple[str, set]:
    """Find French region/department phrases; return (address with them removed, {canonical region tokens})."""
    comps = {re.sub(r"[^a-z0-9]+", " ", c).strip() for c in s_ascii.split(",")}
    found = set()
    for phrase, region in FR_PHRASES:
        if phrase in FR_AMBIGUOUS and phrase not in comps:
            continue
        pat = r"(?<![a-z0-9])" + r"[\s\-']+".join(map(re.escape, phrase.split())) + r"(?![a-z0-9])"
        if re.search(pat, s_ascii):
            found.add("frreg_" + region.replace(" ", "_"))
            s_ascii = re.sub(pat, " ", s_ascii)
    return s_ascii, found


def norm_addr(raw: str) -> dict:
    native_states = set()
    raw = raw or ""
    for native, canon in IN_NATIVE.items():
        if native in raw:
            native_states.add(canon)
            raw = raw.replace(native, " ")
    s = to_ascii(raw)
    s, fr_regions = french_regions(s)
    s = s.replace("##", " ").replace("#", " ")
    toks = []
    for t in _tokens(s):
        t = ADDR_ABBR.get(t, t)
        toks.extend(t.split())
    # expand 2-letter state codes only when they are a whole component (avoid "in", "or" words)
    comps = [c.strip() for c in to_ascii(raw).split(",")]
    states = set()
    for c in comps:
        c2 = c.replace(".", "").strip()
        if c2 in US_STATES:
            states.add(US_STATES[c2])
        elif c2 in IN_STATES:
            states.add(IN_STATES[c2])
    states |= fr_regions | native_states
    clean = [t for t in toks if t not in ADDR_JUNK]
    all_nums = _DIGITS.findall(s)
    nums = sorted(set(all_nums))
    words = [t for t in clean if not t.isdigit()]
    return {
        "addr_clean": " ".join(clean + sorted(states)),
        "addr_words": " ".join(sorted(set(words))),
        "addr_skel": " ".join(sorted({skeleton(t) for t in words if len(t) > 1})),
        "addr_nums": " ".join(nums),
        "addr_first_num": all_nums[0] if all_nums else "",
        "addr_empty": int(not clean),
    }


def normalize_record(name: str, addr: str) -> dict:
    return {**norm_name(name), **norm_addr(addr)}
