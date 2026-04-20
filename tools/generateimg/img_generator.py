"""
Khmer Printed Text OCR - Synthetic Image Generator
====================================================
Input modes
-----------
  (A) No --input flag      ->  stream from HuggingFace  (default)
  (B) --input file.txt     ->  plain UTF-8 text file
  (C) --input folder/      ->  all .txt files inside folder
  (D) --input data.csv     ->  CSV  (auto-detect col or set --input_col)
  (E) --input data.json    ->  JSON array of strings or objects
  (F) --input data.jsonl   ->  JSON-Lines file

Split modes  (--split_mode)
---------------------------
  line   short phrases 3-25 chars  [DEFAULT, best for OCR]
  word   single Khmer words/tokens
  both   generate line AND word images in one run

Mixed content  (--inject_mixed)
--------------------------------
  Randomly injects English words, Arabic numbers, and Khmer numerals
  into lines so the model learns to recognise:
    - spaces between tokens
    - English letters (A-Z a-z) mixed with Khmer
    - Arabic digits (0-9) and Khmer digits (០-៩)
    - common Khmer-news terms  (COVID, km, USD, WHO, GDP …)

Requirements: pip install datasets pillow numpy tqdm
"""

import argparse, csv, json, random, re, unicodedata
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────
KHMER_RANGE = (0x1780, 0x17FF)

DEFAULT_CONFIG = {
    "font_sizes"  : [24, 28, 32, 36, 40],
    "padding_x"   : 12,
    "padding_y"   : 8,
    "bg_colors"   : [
        (255, 255, 255),
        (245, 245, 240),
        (230, 230, 220),
        (255, 252, 235),
    ],
    "text_colors" : [
        (0,   0,   0),
        (20,  20,  20),
        (30,  30,  80),
        (60,  10,  10),
    ],
    "max_line_chars" : 25,
    "min_line_chars" : 3,
}

_COMMON_COLS = ["text","content","sentence","khmer","body","article","data"]

# ── RAQM / TEXT SHAPING ───────────────────────────────────
# Khmer requires a proper shaping engine (libraqm) so that
# subscripts (coeng), vowel signs, and stacked glyphs render
# in the correct visual positions.  PIL supports this via RAQM.
try:
    from PIL import features as _pil_features
    _HAS_RAQM = _pil_features.check_feature("raqm")
except Exception:
    _HAS_RAQM = False

if _HAS_RAQM:
    _LAYOUT = ImageFont.Layout.RAQM
    print("  shaping engine : libraqm  (Khmer glyphs will render correctly)")
else:
    _LAYOUT = ImageFont.Layout.BASIC
    print("  WARNING: libraqm not found — Khmer text will render incorrectly!")
    print("  Fix:")
    print("    Ubuntu/Debian : sudo apt-get install libraqm-dev")
    print("    macOS         : brew install harfbuzz fribidi libraqm")
    print("    then          : pip install --force-reinstall Pillow")
    print()



# ── MIXED CONTENT POOLS ───────────────────────────────────
# English words/acronyms that commonly appear in Khmer news
ENGLISH_WORDS = [
    # orgs / institutions
    "WHO", "UN", "ASEAN", "IMF", "WTO", "ADB", "UNESCO", "UNICEF",
    "NGO", "CEO", "COP", "G20", "NATO", "FBI", "CIA",
    # measurements / units
    "km", "kg", "km²", "ha", "GHz", "MHz", "MB", "GB", "TB",
    "kW", "MW", "GW", "USD", "KHR", "EUR", "GDP", "GNP",
    # tech / media
    "COVID", "COVID-19", "PCR", "AI", "IT", "ICT", "5G", "4G",
    "Wi-Fi", "Internet", "Facebook", "YouTube", "Google", "App",
    "QR", "KHQR", "ATM", "PIN", "OTP", "SMS", "GPS",
    # common English loanwords in Khmer press
    "Online", "Offline", "Email", "Website", "Platform",
    "Minister", "President", "Senator", "General",
    "Report", "Budget", "Project", "Center", "Market",
    # months / days (used in Khmer news dates)
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    # mixed-case short words
    "No.", "Vol.", "p.", "pp.", "vs.", "etc.",
]

# Arabic digit strings (1-4 digits, with common formats)
def _random_arabic_number():
    style = random.choice(["int", "float", "percent", "year", "range", "code"])
    if style == "int":      return str(random.randint(1, 9999))
    if style == "float":    return f"{random.uniform(0.1, 999.9):.1f}"
    if style == "percent":  return f"{random.randint(1, 100)}%"
    if style == "year":     return str(random.randint(2000, 2025))
    if style == "range":    return f"{random.randint(1,50)}-{random.randint(51,100)}"
    if style == "code":     return f"{random.randint(100,999)}-{random.randint(1000,9999)}"
    return str(random.randint(0, 999))

# Khmer digit strings  ០-៩
KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
def _random_khmer_number():
    length = random.randint(1, 4)
    return "".join(random.choice(KHMER_DIGITS) for _ in range(length))

# ── MIXED INJECTION ───────────────────────────────────────
def inject_mixed(text: str, rate: float = 0.4) -> str:
    """
    Randomly insert English words / Arabic numbers / Khmer numbers
    into a Khmer text line.  rate = probability per line of injection.

    Strategy
    --------
    Split on spaces -> randomly replace or append tokens.
    Always preserves at least one Khmer token so the line is still Khmer.
    """
    if random.random() > rate:
        return text                     # leave this line untouched

    tokens = text.split(" ")

    # Choose what to inject
    inject_type = random.choice(["english", "arabic", "khmer_num", "mixed"])

    def _new_token():
        if inject_type == "english":
            return random.choice(ENGLISH_WORDS)
        if inject_type == "arabic":
            return _random_arabic_number()
        if inject_type == "khmer_num":
            return _random_khmer_number()
        # mixed: one of each
        return random.choice([
            random.choice(ENGLISH_WORDS),
            _random_arabic_number(),
            _random_khmer_number(),
        ])

    # How many injections (1 or 2)
    n_inject = random.randint(1, min(2, max(1, len(tokens) // 2)))

    for _ in range(n_inject):
        pos = random.randint(0, len(tokens))
        tokens.insert(pos, _new_token())

    return " ".join(tokens)



# ── KHMER TEXT VALIDATOR ──────────────────────────────────
#
# Khmer combining characters — these CANNOT appear at the
# start of a line or after a split boundary:
#
#   U+17B6-U+17C5  dependent vowel signs  (attach to base)
#   U+17C6-U+17C8  above/below diacritics (attach to base)
#   U+17C9-U+17D1  robat, musication marks
#   U+17D2          COENG  ្  (subscript marker — MUST precede a consonant)
#
# Invisible / zero-width characters that corrupt text:
#   U+200B  ZERO WIDTH SPACE
#   U+200C  ZERO WIDTH NON-JOINER
#   U+200D  ZERO WIDTH JOINER
#   U+FEFF  BOM / ZERO WIDTH NO-BREAK SPACE
#
_KHMER_COMBINING = (
    set(range(0x17B6, 0x17D3))   # all dependent vowels + diacritics + coeng
)
_INVISIBLE = {0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD}  # zero-width + soft-hyphen


def _strip_invisible(text: str) -> str:
    """Remove zero-width and invisible Unicode characters."""
    return "".join(ch for ch in text if ord(ch) not in _INVISIBLE)


def _starts_with_combining(text: str) -> bool:
    """True if the first real character is a Khmer combining mark."""
    for ch in text:
        if ch == " ":
            continue
        return ord(ch) in _KHMER_COMBINING
    return False


def _has_orphan_coeng(text: str) -> bool:
    """
    True if U+17D2 (coeng ្) appears at the very end with nothing after it,
    or is followed by a non-consonant — would render as dotted circle.
    """
    COENG = "\u17D2"
    # Khmer consonants: U+1780-U+17A2
    CONSONANTS = set(range(0x1780, 0x17A3))
    for i, ch in enumerate(text):
        if ch == COENG:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not nxt or ord(nxt) not in CONSONANTS:
                return True
    return False


def validate_khmer_line(text: str) -> bool:
    """
    Return True if the line is safe to render without dotted-circle artifacts.
    Checks:
      1. Does not start with a combining/dependent character
      2. Has no orphaned coeng at end of string
      3. Contains at least one base Khmer consonant
    """
    if not text:
        return False
    if _starts_with_combining(text):
        return False
    if _has_orphan_coeng(text):
        return False
    if not any(is_khmer(ch) for ch in text):
        return False
    return True


def sanitize_line(text: str) -> str:
    """
    Clean a single line:
      - strip invisible chars
      - strip leading combining marks until we reach a valid base
      - strip trailing orphan coeng
    """
    text = _strip_invisible(text)
    # Drop leading combining characters
    while text and ord(text[0]) in _KHMER_COMBINING:
        text = text[1:]
    # Drop trailing coeng with no following consonant
    COENG = "\u17D2"
    if text.endswith(COENG):
        text = text[:-1].rstrip()
    return text.strip()

# ── INPUT LOADERS ─────────────────────────────────────────
def _pick_col(headers, user_col):
    if user_col:
        if user_col in headers:
            return user_col
        raise ValueError(f"Column '{user_col}' not found. Available: {headers}")
    for c in _COMMON_COLS:
        if c in headers:
            print(f"  auto-detected text column: '{c}'")
            return c
    print(f"  using first column: '{headers[0]}'")
    return headers[0]


def load_huggingface(num_rows=50_000):
    from datasets import load_dataset
    print("Streaming Thareah/khmer_news_corpus from HuggingFace ...")
    ds = load_dataset("Thareah/khmer_news_corpus", split="train", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= num_rows: break
        t = row.get("text", "").strip()
        if t: texts.append(t)
    print(f"  {len(texts):,} rows loaded.")
    return texts


def load_txt(path):
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="ignore") if l.strip()]
    print(f"  {path.name}: {len(lines):,} lines")
    return lines


def load_txt_dir(folder):
    files = sorted(folder.glob("*.txt"))
    if not files: raise FileNotFoundError(f"No .txt files in: {folder}")
    texts = []
    for f in files: texts.extend(load_txt(f))
    return texts


def load_csv(path, text_col):
    texts = []
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        col = _pick_col(list(reader.fieldnames or []), text_col)
        for row in reader:
            v = row.get(col, "").strip()
            if v: texts.append(v)
    print(f"  {path.name}: {len(texts):,} rows (col='{col}')")
    return texts


def load_json(path, text_col):
    data = json.load(open(path, encoding="utf-8", errors="ignore"))
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if not lists: raise ValueError(f"No list in JSON keys: {list(data.keys())}")
        data = lists[0]
    if not data: return []
    if isinstance(data[0], str):
        texts = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    elif isinstance(data[0], dict):
        col = _pick_col(list(data[0].keys()), text_col)
        texts = [str(r.get(col, "")).strip() for r in data if r.get(col)]
    else:
        raise ValueError(f"Unexpected item type: {type(data[0])}")
    print(f"  {path.name}: {len(texts):,} entries")
    return texts


def load_jsonl(path, text_col):
    texts, col = [], None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except: continue
            if isinstance(obj, str):
                if obj: texts.append(obj)
            elif isinstance(obj, dict):
                if col is None: col = _pick_col(list(obj.keys()), text_col)
                v = str(obj.get(col, "")).strip()
                if v: texts.append(v)
    print(f"  {path.name}: {len(texts):,} lines")
    return texts


def load_input_texts(input_path, text_col=None, num_rows=50_000):
    """Master dispatcher. input_path=None -> HuggingFace default."""
    if input_path is None:
        return load_huggingface(num_rows)
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"--input path not found: {p}")
    print(f"Loading local input: {p}")
    if p.is_dir(): return load_txt_dir(p)
    s = p.suffix.lower()
    if s == ".txt":               return load_txt(p)
    if s == ".csv":               return load_csv(p, text_col)
    if s == ".json":              return load_json(p, text_col)
    if s in (".jsonl", ".ndjson"): return load_jsonl(p, text_col)
    print(f"  unknown extension '{s}' -- trying plain-text reader ...")
    return load_txt(p)


# ── TEXT PREPROCESSING ────────────────────────────────────
def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = _strip_invisible(text)               # remove zero-width chars first
    text = re.sub(r"<[^>]+>",   " ", text)     # strip HTML tags
    text = re.sub(r"https?://\S+", " ", text)  # strip URLs
    # keep: Khmer block, Latin, Arabic+Khmer digits, common punct, spaces
    text = re.sub(r"[^ក-៿a-zA-Z0-9០-៩ .,!?;:()%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_khmer(ch):
    return KHMER_RANGE[0] <= ord(ch) <= KHMER_RANGE[1]


# ── SPLIT MODE: LINE ──────────────────────────────────────
def split_line_mode(texts, min_len=3, max_len=25):
    lines   = []
    sent_re = re.compile(r"(?<=[។៕៖ៗ])\s*")

    for para in texts:
        para = clean_text(para)
        if not para: continue

        for sent in sent_re.split(para):
            sent = sent.strip()
            if not sent: continue

            if len(sent) <= max_len:
                if len(sent) >= min_len and any(is_khmer(c) for c in sent):
                    lines.append(sent)
                continue

            tokens  = sent.split(" ")
            current = ""
            for tok in tokens:
                if len(tok) > max_len:
                    if current:
                        if len(current) >= min_len: lines.append(current)
                        current = ""
                    for i in range(0, len(tok), max_len):
                        chunk = sanitize_line(tok[i:i + max_len])
                        if len(chunk) >= min_len and validate_khmer_line(chunk): lines.append(chunk)
                    continue
                candidate = (current + " " + tok).strip() if current else tok
                if len(candidate) <= max_len:
                    current = candidate
                else:
                    if len(current) >= min_len: lines.append(current)
                    current = tok
            if len(current) >= min_len:
                lines.append(current)

    seen, unique = set(), []
    for l in lines:
        l = sanitize_line(l)
        if l not in seen and validate_khmer_line(l):
            seen.add(l); unique.append(l)
    return unique


# ── SPLIT MODE: WORD ──────────────────────────────────────
def split_word_mode(texts, min_len=1):
    words = []
    for para in texts:
        para = clean_text(para)
        if not para: continue
        for token in para.split():
            token = sanitize_line(token.strip("។៕៖,.!?;:()-"))
            if len(token) >= min_len and validate_khmer_line(token):
                words.append(token)
    seen, unique = set(), []
    for w in words:
        if w not in seen: seen.add(w); unique.append(w)
    return unique


def prepare_lines(texts, split_mode, min_len, max_len):
    if split_mode == "line":
        return split_line_mode(texts, min_len, max_len), []
    if split_mode == "word":
        return [], split_word_mode(texts, min_len=1)
    if split_mode == "both":
        return split_line_mode(texts, min_len, max_len), split_word_mode(texts, min_len=1)
    raise ValueError(f"Unknown --split_mode: {split_mode!r}. Choose: line | word | both")


# ── FONTS ─────────────────────────────────────────────────
def load_fonts(font_paths, sizes):
    loaded = []
    for path in font_paths:
        p = Path(path)
        if not p.exists():
            print(f"  font not found, skipping: {path}"); continue
        for sz in sizes:
            try:
                loaded.append((p.stem, sz, ImageFont.truetype(str(p), sz, layout_engine=_LAYOUT)))
                print(f"  ok  {p.name} @ {sz}px")
            except Exception as e:
                print(f"  err {p.name} @ {sz}px -- {e}")
    if not loaded: raise RuntimeError("No usable fonts loaded. Check --fonts paths.")
    return loaded


# ── RENDERING ─────────────────────────────────────────────
def render_line(text, font, bg, fg, px=12, py=8):
    """
    Render one line of text with proper Khmer shaping.

    When libraqm is available (_HAS_RAQM=True), PIL uses HarfBuzz under the
    hood for complex-script shaping — subscript consonants (coeng), stacked
    vowels, and other Khmer glyph clusters all render in the correct positions.

    The key parameters that activate shaping:
      layout_engine=ImageFont.Layout.RAQM   (set at font-load time)
      language="km"                          (tells HarfBuzz to apply Khmer OT rules)
      direction="ltr"                        (Khmer is left-to-right)
    """
    # getbbox also respects the layout engine, so measurement is accurate
    bb = font.getbbox(text, language="km", direction="ltr") if _HAS_RAQM else font.getbbox(text)
    w  = max(bb[2] - bb[0] + px * 2, 1)
    h  = max(bb[3] - bb[1] + py * 2, 1)
    img  = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    if _HAS_RAQM:
        draw.text((px - bb[0], py - bb[1]), text,
                  font=font, fill=fg,
                  language="km", direction="ltr")
    else:
        draw.text((px - bb[0], py - bb[1]), text, font=font, fill=fg)
    return img


# ── AUGMENTATIONS ─────────────────────────────────────────
def augment_image(img):
    def gnoise(i):
        a = np.array(i, dtype=np.float32)
        return Image.fromarray(
            np.clip(a + np.random.normal(0, random.uniform(1, 5), a.shape), 0, 255).astype(np.uint8))
    def gblur(i):
        return i.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
    def rot(i):
        return i.rotate(random.uniform(-1.5, 1.5), fillcolor=i.getpixel((0, 0)), expand=False)
    def sp(i):
        a = np.array(i)
        n = int((a.size // 3) * random.uniform(0.001, 0.005))
        for v in (255, 0):
            r = np.random.randint(0, max(a.shape[0]-1, 1), n)
            c = np.random.randint(0, max(a.shape[1]-1, 1), n)
            a[r, c] = v
        return Image.fromarray(a)
    def grain(i):
        a = np.array(i, dtype=np.float32)
        s = random.uniform(0.02, 0.06)
        return Image.fromarray(
            np.clip(a + np.random.uniform(-255*s, 255*s, a.shape), 0, 255).astype(np.uint8))

    for prob, fn in [(0.6, gnoise), (0.4, gblur), (0.5, rot), (0.3, sp), (0.5, grain)]:
        if random.random() < prob:
            img = fn(img)
    return img


# ── GENERATION LOOP ───────────────────────────────────────
def generate_dataset(line_samples, word_samples, fonts, output_dir,
                     num_samples=None, do_augment=True, do_inject=False,
                     inject_rate=0.4, config=None, seed=42):
    if config is None: config = DEFAULT_CONFIG
    random.seed(seed); np.random.seed(seed)

    out     = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_entries    = []
    global_written = 0
    global_skipped = 0
    next_idx       = 1

    def _render_batch(samples, level, target):
        nonlocal global_written, global_skipped, next_idx

        max_for_level = len(samples) * len(fonts)
        if num_samples is not None:
            # split budget evenly between line and word if both active
            budget = num_samples if not (line_samples and word_samples) else num_samples // 2
            actual = min(budget, max_for_level)
            print(f"  {level}-level: {actual:,} images  "
                  f"(capped at {budget:,} of {max_for_level:,} possible)")
        else:
            actual = max_for_level
            print(f"  {level}-level: {actual:,} images  "
                  f"({len(samples):,} unique texts × {len(fonts)} font combos — ALL)")

        pool = [(s, f) for s in samples for f in fonts]
        random.shuffle(pool)
        pool = pool[:actual]

        desc = f"  {level}"
        for text, (fname, fsize, font) in tqdm(pool, unit="img", desc=desc):

            # optionally inject mixed content
            if do_inject and level == "line":
                text = inject_mixed(text, rate=inject_rate)

            try:
                img = render_line(
                    text, font,
                    random.choice(config["bg_colors"]),
                    random.choice(config["text_colors"]),
                    config["padding_x"],
                    config["padding_y"],
                )
            except Exception:
                global_skipped += 1; continue

            if img.width < 8 or img.height < 8:
                global_skipped += 1; continue

            if do_augment:
                img = augment_image(img)

            fn = f"{next_idx:07d}.png"
            img.save(img_dir / fn, "PNG", optimize=False)
            all_entries.append((fn, text, fname, fsize, level))
            global_written += 1
            next_idx += 1

    if line_samples:
        _render_batch(line_samples, "line", None)

    if word_samples:
        _render_batch(word_samples, "word", None)

    # ── SAVE LABELS ───────────────────────────────────────
    # labels.txt  — PaddleOCR / Tesseract compatible
    with open(out / "labels.txt", "w", encoding="utf-8") as f:
        for fn, txt, _, _, _ in all_entries:
            f.write(f"images/{fn}\t{txt}\n")

    # labels_detail.txt — full metadata per image
    with open(out / "labels_detail.txt", "w", encoding="utf-8") as f:
        for fn, txt, fnt, fsz, lvl in all_entries:
            f.write(json.dumps(
                {"file": f"images/{fn}", "text": txt,
                 "font": fnt, "size": fsz, "level": lvl},
                ensure_ascii=False) + "\n")

    # metadata
    total_possible = (len(line_samples) + len(word_samples)) * len(fonts)
    json.dump({
        "total_images"   : global_written,
        "total_possible" : total_possible,
        "sample_cap"     : num_samples if num_samples is not None else "unlimited",
        "skipped"        : global_skipped,
        "augmented"      : do_augment,
        "inject_mixed"   : do_inject,
        "inject_rate"    : inject_rate if do_inject else 0,
        "line_samples"   : len(line_samples),
        "word_samples"   : len(word_samples),
        "fonts"          : [{"name": n, "size": s} for n, s, _ in fonts],
        "seed"           : seed,
    }, open(out / "metadata.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)

    print(f"\nDone!  {global_written:,} images saved  ({global_skipped} skipped)")
    print(f"  labels        -> {out}/labels.txt")
    print(f"  labels_detail -> {out}/labels_detail.txt")
    print(f"  metadata      -> {out}/metadata.json")


# ── CLI ───────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Khmer OCR Synthetic Image Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Split modes
-----------
  line   short phrases 3-25 chars  [DEFAULT]
  word   single Khmer tokens
  both   line + word in one run

Mixed content injection  (--inject_mixed)
-----------------------------------------
  Adds English words, Arabic numbers, Khmer numerals into line images.
  Controls how often injection happens with --inject_rate (default 0.4 = 40%%).

Examples
--------
  # recommended — line mode + mixed injection
  python khmer_ocr_generator.py \\
      --input ./khmer_news_corpus/ \\
      --fonts fonts/KhmerOS.ttf fonts/Battambang-Regular.ttf \\
      --output ./ocr_data --samples 10000 \\
      --max_len 25 --augment --inject_mixed

  # both levels + mixed injection
  python khmer_ocr_generator.py \\
      --input ./khmer_news_corpus/ \\
      --fonts fonts/KhmerOS.ttf fonts/NotoSansKhmer-VariableFont_wdth,wght.ttf \\
      --output ./ocr_full --split_mode both --samples 10000 \\
      --augment --inject_mixed --inject_rate 0.5
        """
    )

    # INPUT
    g = p.add_argument_group("input")
    g.add_argument("--input", default=None, metavar="PATH",
        help="Local source: .txt, folder/, .csv, .json, .jsonl. "
             "Omit to stream from HuggingFace.")
    g.add_argument("--input_col", default=None, metavar="COL",
        help="Column name for text in CSV/JSON. Auto-detected if omitted.")
    g.add_argument("--num_rows", type=int, default=50_000, metavar="N",
        help="Max HuggingFace rows (ignored when --input set). Default: 50000.")

    # SPLIT
    p.add_argument("--split_mode", default="line",
        choices=["line", "word", "both"],
        help="line (default) | word | both.")

    # MIXED CONTENT
    m = p.add_argument_group("mixed content")
    m.add_argument("--inject_mixed", action="store_true",
        help="Randomly inject English words, Arabic numbers, and Khmer numerals "
             "into line-level images so the model learns spaces + mixed scripts.")
    m.add_argument("--inject_rate", type=float, default=0.4, metavar="RATE",
        help="Fraction of lines that get injection (0.0-1.0). Default: 0.4.")

    # FONTS
    p.add_argument("--fonts", nargs="+", required=True, metavar="FONT",
        help="Paths to Khmer .ttf/.otf font files.")
    p.add_argument("--font_sizes", nargs="+", type=int,
        default=DEFAULT_CONFIG["font_sizes"], metavar="PX",
        help="Font sizes in px. Default: 24 28 32 36 40.")

    # OUTPUT
    p.add_argument("--output",  default="./khmer_ocr_dataset", metavar="DIR")
    p.add_argument("--samples", type=int, default=None, metavar="N",
        help="Max images to generate. Omit to generate ALL possible "
             "(text × font × size) combinations. "
             "Example: --samples 500000")

    # TEXT FILTER
    p.add_argument("--min_len", type=int, default=DEFAULT_CONFIG["min_line_chars"])
    p.add_argument("--max_len", type=int, default=DEFAULT_CONFIG["max_line_chars"],
        help="Max chars per line. Keep <= 25 for clean OCR images.")

    # AUGMENT
    p.add_argument("--augment", action="store_true",
        help="Apply noise/blur/rotation/grain augmentations.")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg["font_sizes"]     = args.font_sizes
    cfg["min_line_chars"] = args.min_len
    cfg["max_line_chars"] = args.max_len

    print("=" * 55)
    print("  Khmer OCR Synthetic Image Generator")
    print(f"  Input       : {args.input or 'HuggingFace (Thareah/khmer_news_corpus)'}")
    print(f"  Split mode  : {args.split_mode}")
    print(f"  Max chars   : {args.max_len}  (min: {args.min_len})")
    print(f"  Mixed inject: {'ON  rate=' + str(args.inject_rate) if args.inject_mixed else 'OFF'}")
    print(f"  Fonts       : {len(args.fonts)} file(s)")

    print(f"  Augment     : {args.augment}")
    print(f"  Output      : {args.output}")
    print(f"  Samples     : {args.samples:,} (capped)" if args.samples else "  Samples     : UNLIMITED (all combinations)")
    print(f"  Shaping     : {'libraqm (correct Khmer)' if _HAS_RAQM else 'BASIC (broken Khmer!) — install libraqm'}")
    print("=" * 55)

    raw = load_input_texts(args.input, args.input_col, args.num_rows)
    print(f"  {len(raw):,} raw texts loaded.")

    print(f"Splitting text (mode={args.split_mode}) ...")
    line_samples, word_samples = prepare_lines(raw, args.split_mode, args.min_len, args.max_len)
    if line_samples: print(f"  {len(line_samples):,} unique line-level samples")
    if word_samples: print(f"  {len(word_samples):,} unique word-level samples")

    print("Loading fonts ...")
    fonts = load_fonts(args.fonts, cfg["font_sizes"])
    print(f"  {len(fonts)} (font x size) combos loaded.")

    print(f"\nGenerating images -> {args.output}")
    generate_dataset(
        line_samples = line_samples,
        word_samples = word_samples,
        fonts        = fonts,
        output_dir   = args.output,
        num_samples  = args.samples,
        do_augment   = args.augment,
        do_inject    = args.inject_mixed,
        inject_rate  = args.inject_rate,
        config       = cfg,
        seed         = args.seed,
    )


if __name__ == "__main__":
    main()