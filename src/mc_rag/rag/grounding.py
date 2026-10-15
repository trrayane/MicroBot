"""Grounding rules: answer language, predefined refusals, 'no answer' marker.

Everything here is deterministic and free: no LLM call is needed to pick the
language of a refusal or to know whether the model found an answer.
"""

from __future__ import annotations

import re

DEFAULT_LANGUAGE = "fr"
# Share of Arabic letters above which a message is considered Arabic. Kept low
# because Arabic questions often embed Latin names ("ما هو hackathon MC ؟").
_ARABIC_RATIO = 0.25
NO_ANSWER_MARKER = "[[NO_ANSWER]]"

_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_WORD_RE = re.compile(r"[a-zà-ÿœ]+", re.IGNORECASE)
_FRENCH_ACCENTS = frozenset("àâçéèêëîïôùûüÿœ")

_FRENCH_WORDS = frozenset(
    "le la les un une des du de est et quand comment où ou qui que quoi pourquoi "
    "quel quelle quels quelles je tu il elle nous vous ils pour dans sur avec "
    "c'est qu'est ce cette ces mon ma mes votre est-ce peut puis-je salut bonjour "
    "merci membres droit droits étudiant étudiants inscription scolarité".split()
)
_ENGLISH_WORDS = frozenset(
    "the a an is are was were and or how when where what who why which can could "
    "do does i you we they it for in on with to of my your our this that there "
    "hello hi next join student students member members rights".split()
)

# Human-readable names used in the LLM prompt.
_LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "fr": "French",
    "en": "English",
}

_REFUSALS: dict[str, str] = {
    "fr": (
        "Je ne trouve pas cette information dans les documents de MC Club. "
        "Essaie de reformuler ta question ou contacte directement le bureau."
    ),
    "en": (
        "I couldn't find this information in the MC Club documents. "
        "Try rephrasing your question or contact the board directly."
    ),
    "ar": (
        "لم أجد هذه المعلومة في وثائق MC Club. "
        "حاول إعادة صياغة سؤالك أو تواصل مباشرة مع المكتب."
    ),
}


def detect_language(text: str) -> str:
    """Return ``"ar"``, ``"fr"`` or ``"en"`` for *text* (default: French)."""
    letters = _LETTER_RE.findall(text)
    if not letters:
        return DEFAULT_LANGUAGE

    arabic = sum(1 for ch in letters if _ARABIC_RE.match(ch))
    if arabic / len(letters) >= _ARABIC_RATIO:
        return "ar"

    lowered = text.lower()
    words = [w for w in _WORD_RE.findall(lowered)]
    # Apostrophe forms such as "c'est" / "l'événement".
    words += re.findall(r"\b\w+'\w+", lowered)
    french = sum(1 for w in words if w in _FRENCH_WORDS)
    english = sum(1 for w in words if w in _ENGLISH_WORDS)
    if any(ch in _FRENCH_ACCENTS for ch in lowered):
        french += 2

    if french == english:
        return DEFAULT_LANGUAGE
    return "fr" if french > english else "en"


def language_name(code: str) -> str:
    """Human-readable language name for *code* (used in LLM prompts)."""
    return _LANGUAGE_NAMES.get(code, _LANGUAGE_NAMES[DEFAULT_LANGUAGE])


def refusal_message(language: str) -> str:
    """Predefined 'I don't know' message in *language* (default if unknown)."""
    return _REFUSALS.get(language, _REFUSALS[DEFAULT_LANGUAGE])


def parse_model_answer(raw: str) -> tuple[str, bool]:
    """Split the model output into ``(clean_text, grounded)``.

    The model is instructed to append :data:`NO_ANSWER_MARKER` when the sources
    do not contain the answer. An empty answer is never grounded.
    """
    text = raw or ""
    grounded = NO_ANSWER_MARKER not in text
    cleaned = text.replace(NO_ANSWER_MARKER, "").strip()
    if not cleaned:
        return "", False
    return cleaned, grounded