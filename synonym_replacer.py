# synonym_replacer.py
import random
import re
from typing import List, Optional, Set

import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

# Model for semantic similarity (uses installed sentence-transformers)
_EMB_MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Minimum cosine similarity required to accept a synonym candidate
DEFAULT_MIN_SIM = 0.66

# Quick rule-based academic substitutions (high-precision fallback)
RULE_BASED_MAP = {
    "show": ["demonstrate", "indicate", "reveal"],
    "use": ["utilize", "employ"],
    "big": ["substantial", "considerable"],
    "get": ["obtain", "acquire"],
    "help": ["facilitate", "assist"],
    "make": ["produce", "generate"],
    "improve": ["enhance", "ameliorate"],
}

# Small curated academic whitelist (lowercase). Expand as you like.
ACADEMIC_WHITELIST = {
    "analyze", "demonstrate", "indicate", "utilize", "employ", "optimize",
    "evaluate", "enhance", "mitigate", "implement", "synthesize", "derive",
    "approximate", "quantify", "correlate", "validate", "investigate",
    "significant", "substantial", "robust", "efficient", "scalable",
    "performance", "accuracy", "reliability", "sustainability"
}

# Utility: single-word check
def _is_single_word(s: str) -> bool:
    return bool(re.match(r'^[A-Za-z\-]+$', s))

def _get_wordnet_synonyms(word: str, pos=None) -> List[str]:
    synonyms = set()
    wn_pos = None
    if pos:
        if pos.startswith("J"):
            wn_pos = wordnet.ADJ
        elif pos.startswith("V"):
            wn_pos = wordnet.VERB
        elif pos.startswith("N"):
            wn_pos = wordnet.NOUN
        elif pos.startswith("R"):
            wn_pos = wordnet.ADV

    for syn in wordnet.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ")
            if cand.lower() != word.lower() and _is_single_word(cand):
                synonyms.add(cand)
    return list(synonyms)

def select_best_synonym(original: str, candidates: List[str], min_sim: float = DEFAULT_MIN_SIM) -> Optional[str]:
    if not candidates:
        return None
    orig_emb = _EMB_MODEL.encode(original, convert_to_tensor=True)
    cand_embs = _EMB_MODEL.encode(candidates, convert_to_tensor=True)
    sims = util.cos_sim(orig_emb, cand_embs)[0].cpu().tolist()
    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
    if sims[best_idx] >= min_sim:
        return candidates[best_idx]
    return None

def domain_allowed(word: str, candidate: str, whitelist: Optional[Set[str]] = None) -> bool:
    """
    Accept candidate if in whitelist (if provided) or if candidate is in RULE_BASED_MAP values.
    """
    if not candidate:
        return False
    cw = candidate.lower()
    if whitelist:
        return cw in whitelist
    # allow if candidate is one of the rule-based targets
    for v in RULE_BASED_MAP.values():
        if cw in [x.lower() for x in v]:
            return True
    # final fallback: allow if candidate contains typical academic suffix/prefix heuristics
    return cw.endswith(("ize", "ise", "ate", "ify")) or len(cw) > 5

def replace_synonyms_in_sentence(
    sentence: str,
    p_replace: float = 0.25,
    min_sim: float = DEFAULT_MIN_SIM,
    whitelist: Optional[Set[str]] = None
) -> str:
    """
    Replace select content words in sentence with appropriate academic synonyms.
    p_replace: probability to attempt replacement per eligible token
    whitelist: a set of allowed academic words (lowercase). If None, use heuristic.
    """
    from spacy import load
    nlp = load("en_core_web_sm")
    doc = nlp(sentence)
    replacements = []
    for token in doc:
        if token.is_stop or token.is_punct or token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV"}:
            continue
        if random.random() > p_replace:
            continue
        word = token.text
        # rule-based quick map first
        if word.lower() in RULE_BASED_MAP:
            cand_list = RULE_BASED_MAP[word.lower()]
            chosen = random.choice(cand_list)
            if domain_allowed(word, chosen, whitelist):
                replacements.append((token.idx, token.idx + len(word), chosen))
                continue

        # ask WordNet for candidates
        wn_cands = _get_wordnet_synonyms(word, token.tag_)
        if not wn_cands:
            continue
        best = select_best_synonym(word, wn_cands, min_sim)
        if best and domain_allowed(word, best, whitelist):
            # preserve capitalization
            if word[0].isupper():
                best = best.capitalize()
            replacements.append((token.idx, token.idx + len(word), best))

    # apply replacements in reverse order to keep indices valid
    out = sentence
    for start, end, rep in reversed(replacements):
        out = out[:start] + rep + out[end:]
    return out
