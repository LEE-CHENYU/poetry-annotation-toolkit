# Poetry Annotation Toolkit

A two-source annotation pipeline for poetry analysis, combining Whisper speech recognition timestamps with LLM-based linguistic annotation across 10 dimensions.

Developed for analyzing T.S. Eliot's *The Waste Land* Section III: The Fire Sermon.

## Overview

This toolkit provides two annotation approaches:

1. **Context-Aware Annotation** (`whisper_annotate.py`) - Uses full poem context, line numbers, and surrounding words for nuanced annotation
2. **Context-Agnostic Annotation** (`whisper_annotate_agnostic.py`) - Annotates words in isolation for comparison/baseline

Both scripts use a **two-source approach**:
- **Display text**: Original poem text (correct spelling preserved)
- **Timestamps**: From Whisper speech recognition
- **Matching**: SequenceMatcher alignment between Whisper tokens and original text

## The 10-Axis Annotation Framework

Words are annotated on 10 linguistic dimensions, organized into three layers:

### I. Phonetic Layer (Sound)
*How the word feels in the mouth and ear during performance.*

| Axis | −1 | 0 | +1 |
|------|-----|---|-----|
| **P1. Vowel Openness** | Closed (i, u): "broken", "fingers", "leaf" | Mid (e, o) | Open (a, ɑ, ɔ): "clutch", "bank" |
| **P2. Sonority / Flow** | Obstructed (stops, clusters): "bottles", "cardboard" | Neutral | Liquid (l, r, w): "Sweet Thames", "run softly" |
| **P3. Articulatory Effort** | Easy (glides naturally): "waits", "sweet" | Neutral | Strained/explosive: "back", "desk", "clutch" |

**Key insight**: The Sonority axis maps directly to *The Waste Land*'s central dialectic—River (flow, life) vs. Drought (obstruction, sterility).

### II. Semantic Layer (Meaning Orientation)
*Not "what it means" but how the word orients toward presence, body, time, action.*

| Axis | −1 | 0 | +1 |
|------|-----|---|-----|
| **S1. Embodiment** | Abstract: "memory", "thought" | Neutral | Bodily/sensory: "belly", "slimy", "dragging" |
| **S2. Agency** | Passive/dissolved: "unreproved", "undesired", "bored" | Neutral | Active/volitional: "Endeavours", "engage" |
| **S3. Presence** | Absent/departed: "departed", "empty", "lost" | Neutral | Present/immediate: "here", "now", "wet" |
| **S4. Temporality** | Timeless/mythic: "Tiresias", "foretold", "always" | Continuous: "waiting", "throbbing" | Instant/momentary: "crack", "now", "sudden" |
| **S5. Fragment ↔ Continuum** | Broken/isolated: "Shantih", "HURRY UP", proper nouns | Neutral | Flowing/connected: "and", "while", refrains |

**Key insight**: Eliot's modernist horror isn't violence but the dissolution of agency. People don't act—they "wait", "throb", "are bored". The typist encounter is pure passivity masked as action.

### III. Rhetorical Layer (Voice Structure)
*How the word functions in Eliot's modernist collage. Who speaks? At what register?*

| Axis | −1 | 0 | +1 |
|------|-----|---|-----|
| **R1. Voice Ownership** | Borrowed (Dante/Wagner/scripture): "Sweet Thames" (Spenser), "Shantih" (Upanishads) | Anonymous/collective: "The typist", "the sailor" | Speaker-owned: "I Tiresias" |
| **R2. Register** | Colloquial/vernacular: "HURRY UP PLEASE ITS TIME", "Ta ta" | Documentary/neutral: "cardboard boxes", "cigarette ends" | Lyrical/ritual: "Sweet Thames, run softly", "violet hour" |

**Key insight**: *The Waste Land* has NO stable "I" to carry emotion. Voices are borrowed, fragmented, displaced. What sentiment belongs to a speaker who is Tiresias+typist+Dante+Buddha? This framework tracks that radical instability.

## Why Not Sentiment Analysis?

Traditional sentiment analysis fails for modernist poetry because:

1. **No stable speaker**: The poem's "I" shifts between Tiresias, the typist, Dante, Buddha, and anonymous voices
2. **Collage structure**: Meaning emerges from juxtaposition, not individual word sentiment
3. **Ironic distance**: Words like "Sweet Thames" are simultaneously beautiful AND ironic (the river is polluted)
4. **Register collision**: Sacred and profane occupy the same line—sentiment cannot capture this tension

Our 10-axis framework captures **how words function** in the poem's structure rather than **what they feel**.

## Installation

```bash
pip install openai python-dotenv
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Context-Aware Annotation

```bash
python whisper_annotate.py
```

Options:
- `--force`: Re-process even if progress files exist
- `--section N`: Process only section N

### Context-Agnostic Annotation

```bash
python whisper_annotate_agnostic.py
```

Options:
- `--force`: Re-process all batches
- `--batch N`: Process only batch N

## Input Requirements

Both scripts expect:

1. `text_sections.json` - Section structure with:
   ```json
   {
     "section_num": 1,
     "audio_file": "audio/III. 1.m4a",
     "lines": ["The river's tent is broken...", ...],
     "start_line": 1
   }
   ```

2. Audio files in the specified paths

## Output

- `poetry-data.js` / `poetry-data-agnostic.js` - JavaScript module with annotated words
- `poetry-data-annotated.json` - JSON format
- Progress checkpoints in `annotation_progress/` or `annotation_progress_agnostic/`

Each word includes:
```json
{
  "text": "river's",
  "start": 0.52,
  "end": 0.98,
  "line": 1,
  "audioFile": "audio/III. 1.m4a",
  "vowelOpenness": 0.23,
  "sonority": 0.67,
  "effort": -0.15,
  "embodied": 0.45,
  "agency": -0.32,
  "presence": 0.18,
  "temporality": -0.56,
  "fragment": 0.71,
  "voiceOwnership": 0.08,
  "register": 0.54
}
```

## Two-Source Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Original Text     │     │   Whisper Audio     │
│   (correct spelling)│     │   (timestamps)      │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           ▼                           ▼
    extract_original_tokens()    extract_whisper_tokens()
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
           align_and_transfer_timestamps()
           (SequenceMatcher + contraction handling)
                       │
                       ▼
           interpolate_missing_timestamps()
                       │
                       ▼
               annotate_tokens()
               (GPT-5.2 / GPT-4o)
                       │
                       ▼
              poetry-data.js
```

This solves the problem of Whisper mis-transcribing foreign words (French), proper nouns, and archaic spellings while preserving accurate audio timestamps.

## Sample Annotation Data

The `data_context_aware/` and `data_context_agnostic/` folders contain pre-annotated data for *The Waste Land* Section III.

## Citation

If you use this toolkit or methodology, please cite:

```
Poetry Annotation Toolkit: A 10-Axis Framework for Modernist Poetry Analysis
https://github.com/LEE-CHENYU/poetry-annotation-toolkit
```

## License

MIT License

## Acknowledgments

- T.S. Eliot's *The Waste Land* (1922)
- OpenAI Whisper for speech recognition
- OpenAI GPT for linguistic annotation
