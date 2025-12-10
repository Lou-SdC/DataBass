"""
Fonctions pour traduire une note musicale dans différentes langues.
Usage:
  from multilangual import translate
  translate("C#", "Français")   -> "do♯"
  translate("do", "Anglais")    -> "C"
  translate("1", "Français")    -> "do"
  translate("Rébemol", "Anglais")-> "D♭"
"""

from typing import Tuple
import unicodedata
import re

# Helper: remove accents and normalize
def _normalize_text(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower()

# Canonical mapping: degree -> canonical names (we use degree 1..7 mapped to C..B)
DEGREE_TO_LETTER = {
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "A",
    7: "B",
}

# Map from many possible input names to canonical letter and degree
INPUT_NAME_TO_LETTER = {}

# Solfège (français and many latin languages)
for deg, solfege in zip(
    [1, 2, 3, 4, 5, 6, 7],
    ["do", "re", "mi", "fa", "sol", "la", "si"]
):
    INPUT_NAME_TO_LETTER[solfege] = DEGREE_TO_LETTER[deg]

# Add common accented/alternative forms (ré -> re, do=ut alt)
INPUT_NAME_TO_LETTER.update({
    "ut": "C",
    "re": "D",
    "mi": "E",
    "fa": "F",
    "sol": "G",
    "la": "A",
    "si": "B",
})

# English/German/Dutch letter names (A-G, H special in German)
for ch in "ABCDEFGH":
    INPUT_NAME_TO_LETTER[ch.lower()] = ch.upper()
# numeric jianpu (1-7)
for i in range(1, 8):
    INPUT_NAME_TO_LETTER[str(i)] = DEGREE_TO_LETTER[i]

# Other scripts/names from the table (Russian, Greek, Hindi, Korean, Japanese, Chinese simplified entries)
LANG_SPECIFIC_SOLFEGE = {
    "rus": ["до", "ре", "ми", "фа", "соль", "ля", "си"],
    "greek": ["ντο", "ρε", "μι", "φα", "σολ", "λα", "σι"],
    "hindi": ["स", "रे", "ग", "म", "प", "ध", "नि"],
    "korean": ["다", "라", "마", "바", "사", "가", "나"],
    "japanese": ["ハ", "ニ", "ホ", "ヘ", "ト", "イ", "ロ"],
    # Chinese (jianpu numbers already handled). Add Chinese char names:
    "chinese_chars": ["婷", "涵", "雯", "玲", "琪", "芳", "欣"],
    # Chinese pentatonic (wusheng) mapping for 5-tone set
    "wusheng": ["宮", "商", "角", "徵", "羽"]
}
# incorporate into input map (lowered/normalized)
for key_list in LANG_SPECIFIC_SOLFEGE.values():
    for i, name in enumerate(key_list, start=1):
        INPUT_NAME_TO_LETTER[_normalize_text(name)] = DEGREE_TO_LETTER.get(i if i<=7 else 1)

# Semitone mapping for canonical letters in C major
LETTER_TO_SEMITONE = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11, "H": 11
}

# Reverse: from semitone to "closest" natural letter (prefer C D E F G A B)
SEMITONE_TO_LETTER = {v: k for k, v in LETTER_TO_SEMITONE.items()}

# Target-language dictionaries for the natural notes (no accidentals)
LANG_DICTS = {
    # Keys recognized: French names (user likely uses French); accept English names too.
    "francais":    ["do", "re", "mi", "fa", "sol", "la", "si"],
    "français":    ["do", "ré", "mi", "fa", "sol", "la", "si"],
    "portugais":   ["do", "ré", "mi", "fa", "sol", "lá", "si"],
    "italien":     ["do", "re", "mi", "fa", "sol", "la", "si"],
    "espagnol":    ["do", "re", "mi", "fa", "sol", "la", "si"],
    "neerlandais_be": ["do", "re", "mi", "fa", "sol", "la", "si"],
    "roumain":     ["do", "re", "mi", "fa", "sol", "la", "si"],
    "russe":       ["до", "ре", "ми", "фа", "соль", "ля", "си"],
    "grec":        ["ντο", "ρε", "μι", "φα", "σολ", "λα", "σι"],
    "allemand":    ["C", "D", "E", "F", "G", "A", "H"],
    "anglais":     ["C", "D", "E", "F", "G", "A", "B"],
    "neerlandais": ["C", "D", "E", "F", "G", "A", "B"],  # Netherlands: same as English letters
    "chinois_jianpu": ["1", "2", "3", "4", "5", "6", "7"],
    "chinois_chars":  ["婷", "涵", "雯", "玲", "琪", "芳", "欣"],
    "wusheng":     ["宮", "商", "角", "徵", "羽", None, None],  # pentatonic; 6-7 not used
    "hindi":       ["स", "रे", "ग", "म", "प", "ध", "नि"],
    "korean":      ["다", "라", "마", "바", "사", "가", "나"],
    "japonais":    ["ハ", "ニ", "ホ", "ヘ", "ト", "イ", "ロ"],
}

# Aliases for target_lang names (allow french/english)
LANG_ALIASES = {
    "français": "français",
    "francais": "français",
    "french": "français",
    "anglais": "anglais",
    "english": "anglais",
    "allemand": "allemand",
    "german": "allemand",
    "russe": "russe",
    "russian": "russe",
    "grec": "grec",
    "greek": "grec",
    "italien": "italien",
    "italian": "italien",
    "espagnol": "espagnol",
    "spanish": "espagnol",
    "portugais": "portugais",
    "portuguese": "portugais",
    "neerlandais_be": "neerlandais_be",
    "neerlandais": "neerlandais",
    "dutch": "neerlandais",
    "roumain": "roumain",
    "chinois_jianpu": "chinois_jianpu",
    "chinois": "chinois_jianpu",
    "chinese": "chinois_jianpu",
    "chinese_chars": "chinois_chars",
    "wusheng": "wusheng",
    "hindi": "hindi",
    "korean": "korean",
    "coréen": "korean",
    "japonais": "japonais",
    "japanese": "japonais",
}

# Merge LANG_DICTS keys into aliases so direct keys also valid
for k in list(LANG_DICTS.keys()):
    if k not in LANG_ALIASES:
        LANG_ALIASES[k] = k

# Parse note string -> (semitone, accidental_str)
def _parse_note_to_semitone(note: str) -> Tuple[int, str]:
    s = note.strip()
    if not s:
        raise ValueError("Note vide")
    s_norm = _normalize_text(s)

    # Detect explicit letter forms A-G or H and optional accidental suffix/prefix
    m = re.match(r'^([a-ghA-GH])\s*([#♯b♭]?)(.*)$', s)
    if m:
        base = m.group(1).upper()
        acc = m.group(2)
        if acc == "" and m.group(3).startswith(("sharp", "is")):
            acc = "#"
        sem = LETTER_TO_SEMITONE.get(base)
        if sem is None:
            raise ValueError(f"Base note inconnue: {base}")
        if acc in ("#", "♯"):
            return (sem + 1) % 12, "#"
        if acc in ("b", "♭"):
            return (sem - 1) % 12, "b"
        return sem, ""

    # Detect numeric (jianpu)
    if s_norm in INPUT_NAME_TO_LETTER:
        letter = INPUT_NAME_TO_LETTER[s_norm]
        # Map letter to semitone
        sem = LETTER_TO_SEMITONE.get(letter)
        if sem is None:
            sem = 0
        return sem, ""

    # Also try to catch solfège with accidental like 'reb' or 're#' or 're bemol'
    m2 = re.match(r'^([a-zA-Z]+)(?:\s*[-_]?\s*(?:b|♭|bemol|bemol|ben|flat|bémol|bemol))?$', s_norm)
    # Instead try simpler: check prefix name and trailing b/#
    base_match = None
    for name, letter in INPUT_NAME_TO_LETTER.items():
        if s_norm.startswith(name):
            base_match = (name, letter)
            rest = s_norm[len(name):]
            # rest may contain 'b', 'bemol', '#', 'sharp'
            if rest.startswith("#") or rest.startswith("♯") or rest.startswith("sharp"):
                return (LETTER_TO_SEMITONE[letter] + 1) % 12, "#"
            if rest.startswith("b") or rest.startswith("♭") or rest.startswith("bemol") or rest.startswith("bemol"):
                return (LETTER_TO_SEMITONE[letter] - 1) % 12, "b"
            return LETTER_TO_SEMITONE[letter], ""
    # If nothing matched, raise
    raise ValueError(f"Impossible d'interpréter la note: {note!r}")

def _semitone_to_degree_and_acc(semitone: int) -> Tuple[str, str]:
    """
    Return a canonical natural letter (C D E F G A B/H) and accidental symbol ("", "#" or "b")
    We choose the natural nearest below (i.e., represent as natural + accidental).
    """
    semitone = semitone % 12
    # Try to find exact natural
    if semitone in SEMITONE_TO_LETTER:
        return SEMITONE_TO_LETTER[semitone], ""
    # If not natural, find a natural that +/-1 equals semitone
    for base_sem, letter in SEMITONE_TO_LETTER.items():
        if (base_sem + 1) % 12 == semitone:
            return letter, "#"
        if (base_sem - 1) % 12 == semitone:
            return letter, "b"
    # fallback
    return "C", ""

def translate(note: str, target_lang: str) -> str:
    """
    Translate a musical note into the requested language.
    - `note`: input like "C#", "do", "réb", "1", "до", etc.
    - `target_lang`: language name (French names or common English aliases accepted)
    Returns translated note as string (including a simple accidental marker if needed).
    Raises ValueError if note or language unknown.
    """
    if not isinstance(note, str) or not isinstance(target_lang, str):
        raise ValueError("Les deux arguments doivent être des chaînes de caractères")

    # Normalize target language
    lang_key = _normalize_text(target_lang)
    lang_key = LANG_ALIASES.get(lang_key, lang_key)
    if lang_key not in LANG_DICTS:
        raise ValueError(f"Langue cible non supportée: {target_lang!r}")

    sem, acc = _parse_note_to_semitone(note)
    base_letter, derived_acc = _semitone_to_degree_and_acc(sem)
    # if parse returned acc explicitly, prefer that
    if acc:
        derived_acc = acc

    # Map base_letter to degree 1..7
    # We want to find index in DEGREE_TO_LETTER values
    degree = None
    for d, L in DEGREE_TO_LETTER.items():
        # consider German H as B equivalent
        if base_letter == "H":
            # treat H as B (degree 7)
            if d == 7:
                degree = d
                break
        if L == base_letter:
            degree = d
            break
    if degree is None:
        degree = 1

    lang_list = LANG_DICTS[lang_key]
    # Some target dicts may have None for some degrees (wusheng), handle gracefully
    try:
        base_name = lang_list[degree - 1]
    except Exception:
        base_name = None

    if base_name is None:
        raise ValueError(f"La langue '{target_lang}' ne fournit pas de nom pour ce degré ({degree})")

    # Format accidental for the language (simple rules)
    if derived_acc == "":
        return base_name
    # For German use 'is' for sharp and 'es' for flat (roughly)
    if lang_key == "allemand":
        if derived_acc == "#":
            # e.g., C -> Cis
            return base_name + "is"
        else:
            # E flat -> Es (note: some exceptions omitted)
            return base_name + "es"
    # For languages using letter names (anglais, neerlandais), append accidental symbol
    if lang_key in ("anglais", "neerlandais"):
        return base_name + ( "#" if derived_acc == "#" else "b")
    # For solfège languages, append ♯/♭
    if derived_acc == "#":
        return base_name + "♯"
    else:
        return base_name + "♭"

if __name__ == "__main__":
    # Quick manual tests
    examples = [
        ("C", "Français"),
        ("C#", "Français"),
        ("Db", "Français"),
        ("do", "Anglais"),
        ("ré", "Anglais"),
        ("1", "Français"),
        ("1", "chinois_jianpu"),
        ("A", "Allemand"),
        ("B", "Allemand"),
        ("H", "Anglais"),
        ("mi", "russe"),
        ("sol#", "japonais"),
        ("G#", "allemand"),
    ]
    for n, lang in examples:
        try:
            print(f"{n} -> {translate(n, lang)} ({lang})")
        except Exception as e:
            print(f"{n} -> ERROR: {e}")
