#!/usr/bin/env python3
"""
Context-Agnostic Re-Annotation Script for The Waste Land III

This script takes the existing context-aware annotations (poetry-data-annotated.json)
and re-annotates ONLY the axis dimensions with GPT in context-agnostic mode.

Timestamps, word text, and all other metadata are preserved from the original.

KEY DIFFERENCE: GPT receives ONLY the word text, no position, no context, no poem info.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
INPUT_JSON = Path("poetry-data-annotated.json")
OUTPUT_JSON = Path("poetry-data-annotated-agnostic.json")
OUTPUT_JS = Path("poetry-data-agnostic.js")
PROGRESS_DIR = Path("annotation_progress_agnostic")

# 10-axis framework (same as context-aware)
AXES_DEFINITIONS = """
Annotate each word on 10 axes. All use -1.0 to +1.0 scale.

═══════════════════════════════════════════════════════════════════════════════
I. PHONETIC LAYER (How the word feels in the mouth and ear during performance)
═══════════════════════════════════════════════════════════════════════════════

1. vowelOpenness: Perceived dominant vowel
   -1.0 = Closed vowels (i, u): "broken" /ə/, "fingers" /ɪ/, "leaf" /i/, "sink" — constriction, withholding
   0.0 = Mid vowels (e, o)
   +1.0 = Open vowels (a, ɑ, ɔ): "clutch" /ʌ/, "bank" /æ/ — expansive, grasping

2. sonority: Flow vs. obstruction
   -1.0 = Obstructed (stops, clusters): "bottles" /t/, "desk" /sk/ — blockage, sterility
   0.0 = Neutral
   +1.0 = Liquid (l, r, w, vowels): "Sweet" /sw/, "softly" /s/ — flow, life, water

3. effort: Articulatory effort
   -1.0 = Easy (glides naturally): "waits" /w/, "Sweet" — effortless
   0.0 = Neutral
   +1.0 = Strained / Broken (explosive, effortful): "back" /k/, "clutch" — pressure, bodily tension

═══════════════════════════════════════════════════════════════════════════════
II. SEMANTIC LAYER (Meaning orientation, not "what it means")
═══════════════════════════════════════════════════════════════════════════════

4. embodied: Physical presence vs. abstraction
   -1.0 = Abstract: "memory", "thought", "meaning" — cognitive, non-sensory
   0.0 = Neutral
   +1.0 = Embodied / Bodily: "belly", "slimy", "dragging" — visceral, tactile, physical motion

5. agency: Active volition vs. passivity
   -1.0 = Passive / Dissolved: "unreproved", "undesired", "bored" — acted upon, negated will
   0.0 = Neutral
   +1.0 = Active / Volitional: "Endeavours", "engage" — willed action (even if hollow)

6. presence: Immediate vs. absent
   -1.0 = Absent / Departed: "departed", "empty", "lost", "broken" (as absence of wholeness)
   0.0 = Neutral
   +1.0 = Present / Immediate: "here", "now", "this", "wet" — immediate sensory presence

7. temporality: Instant vs. eternal
   -1.0 = Timeless / Mythic: "Tiresias", "foretold", "always", "death" — eternal, prophetic
   0.0 = Continuous / Durational: "waiting", "throbbing", "drifting" — stretched time
   +1.0 = Instant / Momentary: "crack", "now", "sudden", "violet hour" — specific moment

8. fragment: Broken vs. flowing
   -1.0 = Broken / Isolated: Proper nouns, cultural references, interrupted phrases — "Shantih", "HURRY UP"
   0.0 = Neutral
   +1.0 = Flowing / Connected: Conjunctions, flowing syntax, sustained imagery — "and", "while", lyrical refrains

═══════════════════════════════════════════════════════════════════════════════
III. RHETORICAL / STRUCTURAL LAYER (Function in modernist collage)
═══════════════════════════════════════════════════════════════════════════════

9. voiceOwnership: Who speaks?
   -1.0 = Borrowed (Dante/Wagner/scripture): "Sweet Thames" (Spenser), "Shantih" (Upanishads), "HURRY UP" (pub voice)
   0.0 = Anonymous / Collective: "the typist", "the sailor" — no individual voice
   +1.0 = Speaker-Owned: "I Tiresias" — direct (though still masked) voice

10. register: Language level
    -1.0 = Colloquial / Vernacular: "HURRY UP PLEASE ITS TIME", "Ta ta" — pub speech, class-marked
    0.0 = Neutral / Documentary: "cardboard boxes", "cigarette ends" — catalogue, itemized
    +1.0 = Lyrical / Ritual: "Sweet Thames, run softly", "Shantih", "violet hour" — elevated, ritual language

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: You are annotating words in COMPLETE ISOLATION.
You do NOT know:
- The poem title or author
- What comes before or after this word
- The line number or position in text
- Any contextual meaning

Annotate based ONLY on the word itself as an isolated lexical item.
═══════════════════════════════════════════════════════════════════════════════
"""

def load_context_aware_data(input_path: Path) -> Dict[str, Any]:
    """Load the existing context-aware annotated data."""
    print(f"\n{'='*80}")
    print(f"Loading context-aware data from: {input_path}")
    print(f"{'='*80}\n")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data['words'])} words from context-aware annotations")
    return data


def annotate_batch_agnostic(client: OpenAI, words: List[Dict[str, Any]], batch_start: int, batch_end: int) -> List[Dict[str, Any]]:
    """
    Annotate a batch of words in context-agnostic mode.

    Returns list of annotations with index and axis scores.
    """
    # Build minimal token data - ONLY word text and index
    token_data = []
    for idx in range(batch_start, batch_end):
        word = words[idx]
        token_data.append({
            "index": idx,
            "word": word["text"]  # ONLY the word text - NO context!
        })

    # Minimal prompt - NO section context, NO numbered text, NO position guidance
    prompt = f"""You are annotating individual words on 10 axes.

IMPORTANT: Annotate each word INDEPENDENTLY without any context.
You do NOT know the poem, author, position in text, or surrounding words.

Words to annotate:
{json.dumps(token_data, indent=2, ensure_ascii=False)}

{AXES_DEFINITIONS}

Analyze each word in complete isolation. Return ONLY a JSON object with "words" array.
Each item must include the original "index" and all 10 axis scores (as numbers, not strings).

Example format:
{{
  "words": [
    {{
      "index": 0,
      "vowelOpenness": -0.3,
      "sonority": 0.5,
      "effort": 0.2,
      "embodied": 0.1,
      "agency": -0.4,
      "presence": 0.0,
      "temporality": 0.3,
      "fragment": -0.2,
      "voiceOwnership": 0.0,
      "register": 0.1
    }}
  ]
}}
"""

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"  Annotating batch {batch_start}-{batch_end} (attempt {retry_count + 1})...", end=" ", flush=True)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a linguistic analysis AI. Annotate words in complete isolation without any contextual information. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            annotations = result.get("words", [])

            print(f"✓ Got {len(annotations)} annotations")
            return annotations

        except Exception as e:
            retry_count += 1
            print(f"✗ Error: {e}")
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts, skipping batch")
                return []


def save_progress(output_path: Path, data: Dict[str, Any], batch_num: int):
    """Save progress to JSON file."""
    progress_file = PROGRESS_DIR / f"progress_batch_{batch_num:03d}.json"
    PROGRESS_DIR.mkdir(exist_ok=True)

    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Progress saved: {progress_file}")


def annotate_agnostic(input_path: Path, output_json: Path, output_js: Path, batch_size: int = 50):
    """
    Re-annotate existing data in context-agnostic mode.

    Preserves all timestamps and metadata, only updates axis scores.
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load existing context-aware data
    data = load_context_aware_data(input_path)
    words = data["words"]

    # Prepare progress directory
    PROGRESS_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Re-annotating {len(words)} words in context-agnostic mode")
    print(f"Batch size: {batch_size}")
    print(f"{'='*80}\n")

    # Process in batches
    total_batches = (len(words) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(words))

        print(f"\n[Batch {batch_num + 1}/{total_batches}] Words {batch_start}-{batch_end}")

        # Get context-agnostic annotations
        annotations = annotate_batch_agnostic(client, words, batch_start, batch_end)

        # Merge annotations back into existing word data
        for annotation in annotations:
            idx = annotation["index"]
            if idx < len(words):
                # Update ONLY the axis scores, preserve everything else
                words[idx]["vowelOpenness"] = annotation.get("vowelOpenness", 0.0)
                words[idx]["sonority"] = annotation.get("sonority", 0.0)
                words[idx]["effort"] = annotation.get("effort", 0.0)
                words[idx]["embodied"] = annotation.get("embodied", 0.0)
                words[idx]["agency"] = annotation.get("agency", 0.0)
                words[idx]["presence"] = annotation.get("presence", 0.0)
                words[idx]["temporality"] = annotation.get("temporality", 0.0)
                words[idx]["fragment"] = annotation.get("fragment", 0.0)
                words[idx]["voiceOwnership"] = annotation.get("voiceOwnership", 0.0)
                words[idx]["register"] = annotation.get("register", 0.0)

        # Save progress
        save_progress(output_json, data, batch_num)

        # Rate limiting
        if batch_num < total_batches - 1:
            time.sleep(1)

    # Update metadata
    data["metadata"]["pipeline"] = "context-agnostic-reannotation"
    data["metadata"]["generated"] = datetime.now().isoformat()
    data["metadata"]["source_file"] = str(input_path)
    data["metadata"]["annotation_mode"] = "context-agnostic"

    # Save final JSON
    print(f"\n{'='*80}")
    print(f"Saving final output...")
    print(f"{'='*80}\n")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {output_json}")

    # Generate JS file
    js_content = f"""// Context-Agnostic Annotations for The Waste Land III
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Source: Re-annotated from {input_path}

const poetryDataAgnostic = {json.dumps(data, indent=2, ensure_ascii=False)};

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = poetryDataAgnostic;
}}
"""

    with open(output_js, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"✓ Saved: {output_js}")

    # Summary
    print(f"\n{'='*80}")
    print(f"CONTEXT-AGNOSTIC RE-ANNOTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total words: {len(words)}")
    print(f"Annotations mode: Context-agnostic (word-only)")
    print(f"Timestamps: Preserved from context-aware data")
    print(f"Output: {output_json}")
    print(f"        {output_js}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Re-annotate existing data in context-agnostic mode"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_JSON,
        help=f"Input JSON file with context-aware annotations (default: {INPUT_JSON})"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
        help=f"Output JSON file (default: {OUTPUT_JSON})"
    )
    parser.add_argument(
        "--output-js",
        type=Path,
        default=OUTPUT_JS,
        help=f"Output JS file (default: {OUTPUT_JS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for GPT annotation (default: 50)"
    )

    args = parser.parse_args()

    # Check input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print(f"Please run the context-aware annotation first to generate {args.input}")
        return 1

    # Run re-annotation
    annotate_agnostic(args.input, args.output_json, args.output_js, args.batch_size)

    return 0


if __name__ == "__main__":
    exit(main())
