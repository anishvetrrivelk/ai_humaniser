# text_cleaner.py
import re
from typing import List

try:
    import language_tool_python
    TOOL_AVAILABLE = True
    tool = language_tool_python.LanguageTool('en-US')
except Exception:
    TOOL_AVAILABLE = False

# A compact set of high-quality transitions for academic prose.
ACADEMIC_TRANSITIONS = [
    "Moreover", "Additionally", "Furthermore", "Therefore", "Consequently",
    "Nonetheless", "Nevertheless", "In contrast", "In summary", "Thus"
]

TRANSITION_PATTERN = re.compile(
    r'^(?:' + r'|'.join([re.escape(t) for t in ACADEMIC_TRANSITIONS]) + r')[,]?\s+',
    flags=re.IGNORECASE
)

MULTI_TRANSITIONS_PATTERN = re.compile(
    r'^(?:(' + r'|'.join([re.escape(t) for t in ACADEMIC_TRANSITIONS]) + r')[,]?\s*){2,}',
    flags=re.IGNORECASE
)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_redundant_transitions(sentence: str) -> str:
    """
    Remove repeated/stacked transitions at the beginning of a sentence,
    e.g. "Hence, Consequently, ..." -> "Consequently,"
    """
    # Remove multiple transitions at sentence start
    if MULTI_TRANSITIONS_PATTERN.match(sentence):
        # Keep only the last transition token (most recent)
        matches = MULTI_TRANSITIONS_PATTERN.match(sentence).group(0)
        # find last transition
        last = re.findall(r'(' + r'|'.join([re.escape(t) for t in ACADEMIC_TRANSITIONS]) + r')', matches, flags=re.IGNORECASE)
        if last:
            transition = last[-1].capitalize()
            # remove the matched chunk and re-prepend single transition
            sentence = MULTI_TRANSITIONS_PATTERN.sub('', sentence)
            sentence = f"{transition}, {sentence.lstrip()}"
    else:
        # If starts with a transition but followed spontaneously by another connector word, collapse duplicates
        sentence = TRANSITION_PATTERN.sub(lambda m: m.group(0).capitalize(), sentence)
    return sentence

def limit_transitions_in_text(text: str, max_per_paragraph: int = 2) -> str:
    """
    Ensure that each paragraph doesn't start with many transitions.
    Also ensures a paragraph has at most `max_per_paragraph` transition-starting sentences.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    new_paras: List[str] = []
    for p in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', p.strip())
        transition_count = 0
        new_sents: List[str] = []
        for s in sentences:
            s = s.strip()
            # detect if sentence starts with a transition
            if TRANSITION_PATTERN.match(s):
                transition_count += 1
                if transition_count > max_per_paragraph:
                    # strip transition
                    s = TRANSITION_PATTERN.sub('', s).lstrip()
                else:
                    s = remove_redundant_transitions(s)
            new_sents.append(s)
        new_paras.append(' '.join(new_sents))
    return '\n\n'.join(new_paras)

def grammar_cleanup(text: str) -> str:
    """
    High-quality grammar cleanup using LanguageTool if available,
    otherwise a conservative pass that only fixes common punctuation issues.
    """
    text = normalize_whitespace(text)
    if TOOL_AVAILABLE:
        try:
            matches = tool.check(text)
            corrected = language_tool_python.utils.correct(text, matches)
            return normalize_whitespace(corrected)
        except Exception:
            # fallback to basic cleanup
            return basic_punctuation_cleanup(text)
    else:
        return basic_punctuation_cleanup(text)

def basic_punctuation_cleanup(text: str) -> str:
    # Fix common problems like duplicated commas and spaces around punctuation
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+;', ';', text)
    text = re.sub(r'\s+:', ':', text)
    text = re.sub(r'\s*\.\s*\.\s*\.', '...', text)
    return normalize_whitespace(text)

def clean_text_pipeline(text: str) -> str:
    """
    Full cleaning pipeline: remove/reduce redundant transitions, then grammar cleanup.
    """
    text = limit_transitions_in_text(text, max_per_paragraph=2)
    text = grammar_cleanup(text)
    return text
