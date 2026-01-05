#!/usr/bin/env python3
"""
Unified Annotation Script for The Waste Land III

Two-source approach:
1. Parse original section text → canonical tokens (correct spelling)
2. Whisper transcription → timestamps only
3. Align Whisper to original tokens (SequenceMatcher)
4. Transfer timestamps to original tokens
5. Annotate original tokens with GPT (10 axes)
6. Output poetry-data.js with correct text + accurate timestamps
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
TEXT_SECTIONS_PATH = Path("text_sections.json")
AUDIO_DIR = Path("audio")
OUTPUT_JSON = Path("poetry-data-annotated.json")
OUTPUT_JS = Path("poetry-data.js")
PROGRESS_DIR = Path("annotation_progress")

# 10-axis framework (Phonetic × Semantic × Rhetorical)
AXES_DEFINITIONS = """
Annotate each word on 10 axes. All use -1.0 to +1.0 scale.

═══════════════════════════════════════════════════════════════════════════════
I. PHONETIC LAYER (How the word feels in the mouth and ear during performance)
═══════════════════════════════════════════════════════════════════════════════

1. vowelOpenness: Perceived dominant vowel
   -1.0 = Closed vowels (i, u): "broken" /ə/, "fingers" /ɪ/, "leaf" /i/, "sink" — constriction, withholding
   0.0 = Mid vowels (e, o)
   +1.0 = Open vowels (a, ɑ, ɔ): "clutch" /ʌ/, "bank" /æ/ — expansive, grasping

   Example: "The river's tent is broken: the last fingers of leaf / Clutch and sink into the wet bank."
   Why it matters: Open vowels in The Waste Land often appear at moments of loss or exposure (drought, barren land).
   Closed vowels appear in constricted, interior states.

2. sonority: Sound texture (River vs. Drought dialectic)
   -1.0 = Obstructed (stops, clusters): "bottles", "cardboard", "cigarette" — hard /t/, /k/ stops
   +1.0 = Liquid/flowing (l, r, w, vowels): "Sweet Thames", "run softly" — flows like water

   Example: "Sweet Thames, run softly, till I end my song."
   Core tension: River (flow, liquid, life) vs. Drought (obstruction, blockage, sterility).
   This axis directly maps The Waste Land's water/desert dialectic.

3. effort: Articulatory body perception (felt performance, not IPA phonetics)
   -1.0 = Easy, slides out: "waits" glides on /w/, "sweet" slides out effortlessly
   +1.0 = Strained/explosive: "back" /bæk/ hard /k/ stop, "desk" /dɛsk/ cluster /sk/ requires tongue/lip pressure

   Example: "At the violet hour, when the eyes and back / Turn upward from the desk"
   Body perception: "Clutch" feels broken/explosive. "Sweet" slides out effortlessly.
   The mechanization passage uses strained articulation to mirror bodily alienation.

═══════════════════════════════════════════════════════════════════════════════
II. SEMANTIC LAYER (How the word orients toward presence, body, time, action)
═══════════════════════════════════════════════════════════════════════════════

4. embodied: Physical vs abstract (not "what it means" but orientation to body)
   -1.0 = Abstract: "memory", "thought", "meaning" — cognitive, non-sensory
   +1.0 = Bodily/sensory: "belly", "slimy", "dragging" — visceral, tactile

   Example: "A rat crept softly through the vegetation / Dragging its slimy belly on the bank"
   SPECIAL CASE — Mythic/Symbolic: "Tiresias", "Shantih", "Phoenix" are CONCRETE (not abstract) but non-bodily.
   They exist in ritual/mythic space. Annotate high concreteness but note symbolic function.
   The Waste Land's dialectic: Hyper-embodiment (visceral decay) vs. mythic displacement (quoted rituals).

5. agency: Volitional force (Eliot's modernist horror = dissolution of agency)
   -1.0 = Passive/dissolved: "unreproved", "undesired", "bored", "tired" — acted upon, negated will
   +1.0 = Active/volitional: "Endeavours", "engage" — but often hollow in Eliot

   Example: "Endeavours to engage her in caresses / Which still are unreproved, if undesired."
   Critical insight: Many words are grammatically ACTIVE but existentially PASSIVE or NULL.
   The typist encounter is pure passivity masked as action. People don't act — they "wait", "throb", "are bored".

6. presence: Here/now vs absent (Opening gesture: presence as always-already absence)
   -1.0 = Absent/departed: "departed", "empty", "lost", "broken" (as absence of wholeness)
   +1.0 = Present/immediate: "here", "now", "this", "wet" (sensory presence)

   Example: "The nymphs are departed."
   Section III begins with DEPARTURE. Even embodied details (wet bank, brown land) exist in the wake of what's gone.
   Quoted/Displaced: "Sweet Thames" — physically present river but displaced through Spenser quotation.

7. temporality: Time quality (Eliot's mythic method: multiple temporal registers simultaneously)
   -1.0 = Timeless/mythic: "Tiresias" exists across all time. "foretold" — prophetic, eternal return
   0.0 = Continuous/durational: "waiting", "throbbing", "drifting" — stretched time
   +1.0 = Instant/momentary: "crack", "now", "sudden", "violet hour" (specific moment)

   Example: "I Tiresias, old man with wrinkled dugs / Perceived the scene, and foretold the rest—"
   The Waste Land overlays mythic time (Tiresias, Philomel, Buddha) onto modern London NOW.

8. fragment: Eliot's collage structure at word level
   -1.0 = Broken/isolated: Proper nouns, cultural references, interrupted phrases — "Shantih", "HURRY UP"
   +1.0 = Flowing/connected: Conjunctions, flowing syntax, sustained imagery — "and", "while", lyrical refrains

   Words can be syntactically continuous but semantically fragmented.

═══════════════════════════════════════════════════════════════════════════════
III. RHETORICAL LAYER (Voice structure: Who speaks? At what register?)
═══════════════════════════════════════════════════════════════════════════════

9. voiceOwnership: Whose language? (The Waste Land has NO stable "I" to carry emotion)
   -1.0 = Borrowed (Dante/Wagner/scripture): "Sweet Thames" (Spenser), "Shantih" (Upanishads)
   0.0 = Anonymous/collective: "The typist", "the sailor" — collective modern figures, no individual voice
   +1.0 = Speaker-owned: "I Tiresias" — But who IS Tiresias? Both speaker and mythic mask.

   Voices are borrowed, fragmented, displaced. What sentiment belongs to a speaker who is
   Tiresias+typist+Dante+Buddha? This axis tracks that radical instability.

10. register: Social/aesthetic level (Register jumps = Waste Land tension)
    -1.0 = Colloquial/vernacular: "HURRY UP PLEASE ITS TIME" (pub speech), "Ta ta" (class-marked goodbye)
    0.0 = Documentary/neutral: "cardboard boxes", "cigarette ends" (catalogue, itemized)
    +1.0 = Lyrical/ritual/elevated: "Sweet Thames, run softly", "violet hour", "Shantih"

    Within ONE stanza, Eliot moves from ritual refrain to trash inventory.
    High and low collapse. Sacred and profane occupy the same line.

═══════════════════════════════════════════════════════════════════════════════
ANNOTATION GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

CRITICAL INSTRUCTIONS:
- These are ORIGINAL TEXT TOKENS from section text (correct spelling preserved)
- Annotate each word in context of surrounding words, line position, and cultural frame
- Eliot's method: fragments gain meaning through juxtaposition
- Consider the speaker, register shifts, and mythic/modern layering

VALUE DISTRIBUTION:
- Use the FULL range from -1.0 to +1.0 with fine gradations
- AVOID clustering at round values (0.0, 0.5, -0.5) - these should be rare
- Most words should have varied, nuanced values like 0.23, -0.35, 0.72, -0.18, 0.86
- Only use exactly 0.0 when a word is truly neutral on that axis
- Each word is unique - values should reflect its specific character in context
- Consider subtle differences: -0.15 vs -0.32 vs -0.67 are meaningfully different

Return ONLY a JSON object with "words" array:
{
  "words": [
    {
      "index": 0,
      "vowelOpenness": 0.47, "sonority": -0.28, "effort": 0.15,
      "embodied": 0.83, "agency": -0.52, "presence": 0.11, "temporality": -0.74, "fragment": 0.38,
      "voiceOwnership": 0.06, "register": 0.62
    }
  ]
}
"""

AXIS_NAMES = [
    "vowelOpenness", "sonority", "effort",
    "embodied", "agency", "presence", "temporality", "fragment",
    "voiceOwnership", "register"
]


@dataclass
class Token:
    """Represents a token from original text with timestamps from Whisper."""
    text: str              # Original spelling (from section text)
    audio_file: str
    line: int = 0
    pos_in_line: int = 0   # Position within line (1-indexed)
    start: float = 0.0     # From Whisper alignment
    end: float = 0.0       # From Whisper alignment
    timestamp_source: str = "estimated"  # "whisper" or "estimated"
    token_index: int = 0
    # Annotations (filled by GPT)
    vowelOpenness: float = 0.0
    sonority: float = 0.0
    effort: float = 0.0
    embodied: float = 0.0
    agency: float = 0.0
    presence: float = 0.0
    temporality: float = 0.0
    fragment: float = 0.0
    voiceOwnership: float = 0.0
    register: float = 0.0


@dataclass
class WhisperToken:
    """Represents a raw token from Whisper (for timestamp extraction only)."""
    text: str
    start: float
    end: float


# Contraction handling for alignment (from merge_timestamps.py)
SUFFIXES = {"s", "m", "ll", "re", "ve", "t", "d"}
PREFIXES = {"d", "l", "j", "t", "m", "s", "c"}
PRONOUN_D_SUFFIX = {"i", "he", "she", "we", "you", "they", "it", "who", "that"}


def normalize_word(word: str) -> str:
    """Normalize word for comparison (strip punctuation, lowercase)."""
    word = word.lower().replace("'", "'").replace(""", '"').replace(""", '"')
    word = re.sub(r"[^a-z0-9]+", "", word)  # Remove all non-alphanumeric
    return word


def load_sections() -> List[dict]:
    """Load section structure from text_sections.json."""
    if not TEXT_SECTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {TEXT_SECTIONS_PATH}")

    with open(TEXT_SECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_original_tokens(lines: List[str], start_line: int, audio_file: str) -> List[Token]:
    """
    Parse original section text into canonical tokens.
    These have correct spelling (unlike Whisper which may mis-transcribe).
    """
    tokens = []
    word_pattern = re.compile(r"[\w']+")

    for line_idx, line in enumerate(lines):
        line_num = start_line + line_idx
        # Skip empty lines but don't skip line numbering
        if not line.strip():
            continue

        words = word_pattern.findall(line)
        for pos, word in enumerate(words, 1):
            tokens.append(Token(
                text=word,  # Original spelling preserved
                audio_file=audio_file,
                line=line_num,
                pos_in_line=pos
            ))

    return tokens


def fuse_tokens_for_matching(tokens: List[Token]) -> List[dict]:
    """
    Fuse adjacent tokens that form contractions for better matching.
    E.g., ["river", "'s"] → ["river's"] for matching against Whisper's "river's"
    """
    fused = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        cur_norm = normalize_word(cur.text)

        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            nxt_norm = normalize_word(nxt.text)

            # Handle "'d" suffix for pronouns (I'd, he'd, etc.)
            if nxt_norm == "d" and cur_norm in PRONOUN_D_SUFFIX:
                fused.append({"norm": cur_norm + nxt_norm, "indices": [i, i + 1]})
                i += 2
                continue

            # Handle common suffixes ('s, 'm, 'll, 're, 've, 't)
            if nxt_norm in SUFFIXES and cur_norm:
                fused.append({"norm": cur_norm + nxt_norm, "indices": [i, i + 1]})
                i += 2
                continue

            # Handle French prefixes (d', l', j', etc.)
            if cur_norm in PREFIXES and nxt_norm:
                fused.append({"norm": cur_norm + nxt_norm, "indices": [i, i + 1]})
                i += 2
                continue

        fused.append({"norm": cur_norm, "indices": [i]})
        i += 1

    return fused


def align_and_transfer_timestamps(original_tokens: List[Token], whisper_tokens: List[WhisperToken]) -> List[Token]:
    """
    Align Whisper tokens to original tokens and transfer timestamps.
    Uses SequenceMatcher for fuzzy matching.
    """
    if not whisper_tokens:
        return original_tokens

    # Fuse original tokens for better matching
    fused = fuse_tokens_for_matching(original_tokens)
    fused_norms = [f["norm"] for f in fused]
    whisper_norms = [normalize_word(w.text) for w in whisper_tokens]

    # Use SequenceMatcher to find best alignment
    matcher = SequenceMatcher(None, fused_norms, whisper_norms, autojunk=False)

    # Build mapping: fused_index -> whisper_index
    mapping = [None] * len(fused)
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            mapping[block.a + i] = block.b + i

    # Second pass: try to match remaining unmatched tokens by position
    used_whisper = {idx for idx in mapping if idx is not None}
    for i, fused_norm in enumerate(fused_norms):
        if mapping[i] is not None or not fused_norm:
            continue
        # Estimate expected position
        expected = round(i / max(len(fused_norms) - 1, 1) * (len(whisper_norms) - 1)) if len(whisper_norms) > 1 else 0
        candidates = [idx for idx, wnorm in enumerate(whisper_norms)
                      if wnorm == fused_norm and idx not in used_whisper]
        if candidates:
            best = min(candidates, key=lambda idx: abs(idx - expected))
            mapping[i] = best
            used_whisper.add(best)

    # Transfer timestamps from Whisper to original tokens
    for fused_idx, whisper_idx in enumerate(mapping):
        if whisper_idx is None:
            continue

        whisper_token = whisper_tokens[whisper_idx]
        indices = fused[fused_idx]["indices"]

        if len(indices) == 1:
            # Single token mapping
            original_tokens[indices[0]].start = whisper_token.start
            original_tokens[indices[0]].end = whisper_token.end
            original_tokens[indices[0]].timestamp_source = "whisper"
        else:
            # Multiple tokens fused - split the time
            duration = whisper_token.end - whisper_token.start
            step = max(duration / len(indices), 0.01)
            for offset, idx in enumerate(indices):
                original_tokens[idx].start = whisper_token.start + step * offset
                original_tokens[idx].end = whisper_token.start + step * (offset + 1)
                original_tokens[idx].timestamp_source = "whisper"

    return original_tokens


def interpolate_missing_timestamps(tokens: List[Token]) -> List[Token]:
    """
    Fill in timestamps for tokens that weren't matched to Whisper.
    Uses linear interpolation between known timestamps.
    """
    # Find tokens with timestamps
    known_indices = [i for i, t in enumerate(tokens) if t.timestamp_source == "whisper"]

    if not known_indices:
        # No timestamps at all - estimate based on section duration
        # Assume ~0.3 seconds per word as fallback
        for i, token in enumerate(tokens):
            token.start = i * 0.3
            token.end = (i + 1) * 0.3
        return tokens

    # Interpolate between known timestamps
    for i, token in enumerate(tokens):
        if token.timestamp_source == "whisper":
            continue

        # Find nearest known timestamps before and after
        before = [k for k in known_indices if k < i]
        after = [k for k in known_indices if k > i]

        if before and after:
            # Interpolate between known points
            b_idx = before[-1]
            a_idx = after[0]
            b_end = tokens[b_idx].end
            a_start = tokens[a_idx].start
            gap = a_start - b_end
            steps = a_idx - b_idx
            step_size = gap / steps
            pos_in_gap = i - b_idx
            token.start = b_end + step_size * (pos_in_gap - 1)
            token.end = b_end + step_size * pos_in_gap
        elif before:
            # After all known - extend from last
            b_idx = before[-1]
            token.start = tokens[b_idx].end + (i - b_idx - 1) * 0.3
            token.end = token.start + 0.3
        else:
            # Before all known - extend backward from first
            a_idx = after[0]
            gap = i - 0
            token.end = tokens[a_idx].start - (a_idx - i - 1) * 0.3
            token.start = token.end - 0.3

    return tokens


def transcribe_audio(client: OpenAI, audio_path: str, vocabulary_hint: str = "") -> dict:
    """Call Whisper API to transcribe audio with word-level timestamps.

    Args:
        client: OpenAI client
        audio_path: Path to audio file
        vocabulary_hint: Optional text prompt with expected vocabulary to improve recognition
                        of proper nouns, foreign words, and unusual spellings.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en",
            prompt=vocabulary_hint if vocabulary_hint else None
        )
    return result


def build_vocabulary_hint(lines: List[str]) -> str:
    """Build a vocabulary hint from section text to improve Whisper recognition.

    Uses the actual section text as the prompt, which helps Whisper recognize:
    - Exact vocabulary and phrasing
    - Proper nouns, foreign words, archaic spellings
    - The overall style and rhythm of the speech

    Whisper uses this as a conditioning signal, not a forced transcript,
    so providing the expected text improves accuracy without degrading it.
    """
    # Use the section text directly - this is what will be spoken
    full_text = " ".join(line.strip() for line in lines if line.strip())

    # Whisper prompt limit is ~224 tokens (~1000 chars for English)
    # Truncate if needed, but prefer to keep as much as possible
    if len(full_text) > 1000:
        full_text = full_text[:1000]

    return full_text


def extract_whisper_tokens(whisper_result) -> List[WhisperToken]:
    """Extract tokens from Whisper response (for timestamps only)."""
    tokens = []
    words = whisper_result.words or []

    for word in words:
        if isinstance(word, dict):
            text = word.get("word", "").strip()
            start = float(word.get("start", 0))
            end = float(word.get("end", 0))
        else:
            text = getattr(word, "word", "").strip()
            start = float(getattr(word, "start", 0))
            end = float(getattr(word, "end", 0))

        if not text:
            continue

        tokens.append(WhisperToken(
            text=text,
            start=start,
            end=end
        ))

    return tokens


def map_to_line_numbers(tokens: List[WhisperToken], original_lines: List[str], start_line: int) -> List[WhisperToken]:
    """
    Map Whisper tokens to original poem line numbers using fuzzy matching.

    Uses SequenceMatcher to align normalized tokens to original text words,
    then assigns line numbers based on matches.
    """
    if not tokens or not original_lines:
        return tokens

    # Build original word index: [(normalized_word, line_num), ...]
    original_words = []
    for i, line in enumerate(original_lines):
        if not line.strip():
            continue
        line_num = start_line + i
        words = re.findall(r"[A-Za-z0-9']+", line)
        for word in words:
            original_words.append((normalize_word(word), line_num))

    if not original_words:
        # Fallback: assign all tokens to start_line
        for token in tokens:
            token.line = start_line
        return tokens

    # Normalize Whisper tokens
    token_norms = [normalize_word(t.text) for t in tokens]
    original_norms = [w[0] for w in original_words]

    # Use SequenceMatcher for alignment
    matcher = SequenceMatcher(None, token_norms, original_norms, autojunk=False)

    # Build mapping: token_index -> original_index
    token_to_original = [None] * len(tokens)
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            token_to_original[block.a + i] = block.b + i

    # Assign line numbers
    last_known_line = start_line
    for i, token in enumerate(tokens):
        if token_to_original[i] is not None:
            token.line = original_words[token_to_original[i]][1]
            last_known_line = token.line
        else:
            # Estimate based on previous known line
            token.line = last_known_line

    return tokens


def annotate_batch(client: OpenAI, all_tokens: List[Token], batch_start: int, batch_end: int, section_text: str, section_info: dict) -> List[dict]:
    """Annotate a batch of tokens with GPT."""
    # Build token context for prompt - use all_tokens for prev/next across batch boundaries
    token_data = []

    for idx in range(batch_start, batch_end):
        token = all_tokens[idx]

        # Use all_tokens for prev/next to cross batch boundaries
        prev_text = all_tokens[idx - 1].text if idx > 0 else ""
        next_text = all_tokens[idx + 1].text if idx + 1 < len(all_tokens) else ""

        token_data.append({
            "index": idx,
            "word": token.text,
            "prev": prev_text,
            "next": next_text,
            "line": token.line,
            "pos_in_line": token.pos_in_line  # Already set by extract_original_tokens
        })

    context = f"Section {section_info.get('section_num', '?')} of The Fire Sermon"
    start_line = section_info.get("start_line", 1)
    original_lines = section_info.get("lines", [])

    # Add line numbers to section text for precise word location
    numbered_text = "\n".join(
        f"{start_line + i}: {line}"
        for i, line in enumerate(original_lines)
    )

    prompt = f"""You are analyzing T.S. Eliot's "The Waste Land" Section III.

Context: {context}

Full section text with line numbers (use for precise word location):
{numbered_text}

IMPORTANT: These are ORIGINAL TEXT TOKENS from the poem (correct spelling preserved).
Each token includes its line number and position within that line (pos_in_line).
Use the numbered text above to verify exact word location.

Words to annotate (use adjacent words as higher-weight context):
{json.dumps(token_data, indent=2, ensure_ascii=False)}

{AXES_DEFINITIONS}

Analyze each word in context. Return ONLY a JSON object with "words" array.
Each item must include the original "index" and all 10 axis scores.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a literary analysis AI specializing in modernist poetry. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        annotations = json.loads(content)

        # Handle different response formats
        if isinstance(annotations, dict):
            if 'words' in annotations:
                return annotations['words']
            elif 'annotations' in annotations:
                return annotations['annotations']
            else:
                for key, value in annotations.items():
                    if isinstance(value, list) and len(value) > 0:
                        return value
        elif isinstance(annotations, list):
            return annotations

        return []

    except Exception as e:
        print(f"    Error calling GPT: {e}")
        return []


def annotate_tokens(client: OpenAI, tokens: List[Token], section_text: str, section_info: dict) -> List[Token]:
    """Annotate all tokens in batches."""
    batch_size = 50

    for i in range(0, len(tokens), batch_size):
        batch_end = min(i + batch_size, len(tokens))
        print(f"    Annotating tokens {i + 1}-{batch_end}...")

        # Pass all tokens so batch can access prev/next across boundaries
        annotations = annotate_batch(client, tokens, i, batch_end, section_text, section_info)

        # Apply annotations to tokens
        for ann in annotations:
            idx = ann.get("index")
            if idx is None or idx < i or idx >= batch_end:
                continue

            token = tokens[idx]
            for axis in AXIS_NAMES:
                setattr(token, axis, ann.get(axis, 0.0))

        # Rate limiting
        time.sleep(0.5)

    return tokens


def fix_zero_durations(tokens: List[Token]) -> List[Token]:
    """Fix words with zero duration (start == end) by spreading them."""
    for i, token in enumerate(tokens):
        if token.end <= token.start:
            # Look for next word's start time
            next_start = None
            if i + 1 < len(tokens):
                next_start = tokens[i + 1].start

            if next_start is not None and next_start > token.start:
                token.end = next_start
            else:
                token.end = token.start + 0.1  # Default 100ms duration
    return tokens


def save_progress(progress_file: Path, tokens: List[Token]):
    """Save progress checkpoint."""
    data = [asdict(t) for t in tokens]
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_progress(progress_file: Path) -> List[Token]:
    """Load progress from checkpoint."""
    with open(progress_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Token(**d) for d in data]


def build_output(all_tokens: List[Token]) -> dict:
    """Build final output structure."""
    words = []
    for token in all_tokens:
        word_entry = {
            "text": token.text,
            "start": token.start,
            "end": token.end,
            "audioFile": token.audio_file,
            "line": token.line,
            # Phonetic
            "vowelOpenness": token.vowelOpenness,
            "sonority": token.sonority,
            "effort": token.effort,
            # Semantic
            "embodied": token.embodied,
            "agency": token.agency,
            "presence": token.presence,
            "temporality": token.temporality,
            "fragment": token.fragment,
            # Rhetorical
            "voiceOwnership": token.voiceOwnership,
            "register": token.register
        }
        words.append(word_entry)

    return {
        "words": words,
        "axes": {
            # Phonetic Layer
            "vowelOpenness": {
                "label": "Vowel Openness",
                "category": "phonetic",
                "min": -1,
                "max": 1,
                "description": "Closed vowels (i, u) vs. open vowels (a, ɑ, ɔ)"
            },
            "sonority": {
                "label": "Sonority / Flow",
                "category": "phonetic",
                "min": -1,
                "max": 1,
                "description": "Obstructed (stops, clusters) vs. liquid/flowing (l, r, w, vowels)"
            },
            "effort": {
                "label": "Articulatory Effort",
                "category": "phonetic",
                "min": -1,
                "max": 1,
                "description": "Easy, natural vs. strained, explosive"
            },
            # Semantic Layer
            "embodied": {
                "label": "Embodiment",
                "category": "semantic",
                "min": -1,
                "max": 1,
                "description": "Abstract vs. embodied/bodily"
            },
            "agency": {
                "label": "Agency",
                "category": "semantic",
                "min": -1,
                "max": 1,
                "description": "Passive/dissolved vs. active/volitional"
            },
            "presence": {
                "label": "Presence",
                "category": "semantic",
                "min": -1,
                "max": 1,
                "description": "Absent/departed vs. present/immediate"
            },
            "temporality": {
                "label": "Temporality",
                "category": "semantic",
                "min": -1,
                "max": 1,
                "description": "Timeless/mythic vs. instant/momentary"
            },
            "fragment": {
                "label": "Fragment / Continuum",
                "category": "semantic",
                "min": -1,
                "max": 1,
                "description": "Broken/interrupted vs. flowing/continuous"
            },
            # Rhetorical Layer
            "voiceOwnership": {
                "label": "Voice Ownership",
                "category": "rhetorical",
                "min": -1,
                "max": 1,
                "description": "Borrowed (cultural) vs. speaker-owned"
            },
            "register": {
                "label": "Register",
                "category": "rhetorical",
                "min": -1,
                "max": 1,
                "description": "Colloquial/vernacular vs. lyrical/ritual"
            }
        },
        "metadata": {
            "title": "The Waste Land - Section III: The Fire Sermon",
            "author": "T.S. Eliot",
            "totalWords": len(words),
            "pipeline": "two-source-unified",
            "annotationVersion": "3.0",
            "annotatedBy": "GPT-5.2",
            "whisperModel": "whisper-1",
            "axisFramework": "Phonetic × Semantic × Rhetorical (10 axes)",
            "generatedAt": datetime.now().isoformat()
        }
    }


def write_output(poetry_data: dict):
    """Write final output files."""
    # Save as JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(poetry_data, f, indent=2, ensure_ascii=False)

    # Save as JavaScript
    js_content = f"""// The Waste Land III - Poetry Data
// Whisper-first unified pipeline
// Generated: {datetime.now().isoformat()}

const poetryData = {json.dumps(poetry_data, indent=2, ensure_ascii=False)};

// Expose globally
if (typeof window !== 'undefined') {{
    window.poetryData = poetryData;
}}

// CommonJS export
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = poetryData;
}}
"""

    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write(js_content)


def main():
    """Main unified annotation pipeline."""
    parser = argparse.ArgumentParser(description="Unified Whisper-first annotation for The Waste Land III")
    parser.add_argument("--force", action="store_true", help="Re-process even if progress files exist")
    parser.add_argument("--section", type=int, help="Process only a specific section number")
    args = parser.parse_args()

    print("=" * 60)
    print("THE WASTE LAND III - WHISPER-FIRST ANNOTATION")
    print("=" * 60)

    # Initialize
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment")

    client = OpenAI(api_key=api_key)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    # Load sections
    print("\n[1] Loading sections...")
    sections = load_sections()
    print(f"    Found {len(sections)} sections")

    # Filter if specific section requested
    if args.section is not None:
        sections = [s for s in sections if s.get("section_num") == args.section]
        if not sections:
            raise SystemExit(f"Section {args.section} not found")

    # Process each section
    all_tokens = []
    global_index = 0

    print("\n[2] Processing sections...")
    for section in sections:
        suffix = section.get("suffix", str(section.get("section_num", "x")))
        audio_file = section.get("audio_file", "")

        print(f"\n  Section {suffix}:")
        print(f"    Audio: {audio_file}")

        progress_file = PROGRESS_DIR / f"progress_{suffix}.json"

        # Check for existing progress
        if progress_file.exists() and not args.force:
            print(f"    Loading from checkpoint...")
            tokens = load_progress(progress_file)
            print(f"    Loaded {len(tokens)} tokens")
        else:
            start_line = section.get("start_line", 1)
            original_lines = section.get("lines", [])

            # Step 1: Extract original tokens (for correct spelling)
            print(f"    Extracting original tokens from section text...")
            tokens = extract_original_tokens(original_lines, start_line, audio_file)
            print(f"    Got {len(tokens)} tokens from original text")

            # Step 2: Whisper transcription (for timestamps only)
            print(f"    Transcribing audio with Whisper...")
            try:
                # Build vocabulary hint from section text to improve recognition
                vocab_hint = build_vocabulary_hint(original_lines)
                whisper_result = transcribe_audio(client, audio_file, vocabulary_hint=vocab_hint)
                whisper_tokens = extract_whisper_tokens(whisper_result)
                print(f"    Got {len(whisper_tokens)} tokens from Whisper")
            except FileNotFoundError as e:
                print(f"    Skipping: {e}")
                continue
            except Exception as e:
                print(f"    Whisper error: {e}")
                continue

            # Step 3: Align and transfer timestamps
            print(f"    Aligning Whisper timestamps to original tokens...")
            tokens = align_and_transfer_timestamps(tokens, whisper_tokens)
            matched = sum(1 for t in tokens if t.timestamp_source == "whisper")
            print(f"    Matched {matched}/{len(tokens)} tokens")

            # Step 4: Interpolate missing timestamps
            print(f"    Interpolating missing timestamps...")
            tokens = interpolate_missing_timestamps(tokens)

            # Step 5: GPT annotation
            section_text = "\n".join(original_lines)
            print(f"    Annotating with GPT-5.2...")
            tokens = annotate_tokens(client, tokens, section_text, section)

            # Save checkpoint
            save_progress(progress_file, tokens)
            print(f"    Saved checkpoint: {progress_file}")

        # Assign global indices
        for token in tokens:
            token.token_index = global_index
            global_index += 1

        all_tokens.extend(tokens)

    # Fix zero-duration timestamps
    print(f"\n[3] Fixing zero-duration timestamps...")
    all_tokens = fix_zero_durations(all_tokens)

    # Build and write output
    print(f"\n[4] Building output ({len(all_tokens)} total tokens)...")
    poetry_data = build_output(all_tokens)
    write_output(poetry_data)

    print(f"\n[5] Output written:")
    print(f"    {OUTPUT_JSON}")
    print(f"    {OUTPUT_JS}")

    print("\n" + "=" * 60)
    print("ANNOTATION COMPLETE!")
    print("=" * 60)
    print(f"\nTotal words: {len(all_tokens)}")
    print(f"Axis framework: 10 axes (Phonetic × Semantic × Rhetorical)")


if __name__ == "__main__":
    main()
