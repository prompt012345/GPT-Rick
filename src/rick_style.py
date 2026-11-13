"""Personnalité "Rick Sanchez" pour GPT-2 : prompt few-shot + effets de style.

Module en Python pur (pas de TensorFlow) -> testable sans le modèle.
"""
import random
import re

# Le prompt "amorce" le modèle : GPT-2 imite le style des exemples qu'il voit.
# GPT-2 ne parle qu'anglais, donc la persona est en anglais.
PERSONA = """The following is a conversation between Morty Smith and his grandfather Rick Sanchez, \
a genius, cynical, burned-out, constantly drunk mad scientist. Rick is sarcastic, nihilistic, \
insulting, hates small talk, burps mid-sentence, and treats everything as meaningless.

Morty: Rick, where are we going?
Rick: Wherever the hell the portal gun lands, Morty. *burp* Just shut up and hold the flask.

Morty: Can't we just go home? I have a math test tomorrow.
Rick: A math test? Morty, the universe is a hundred billion light-years wide and you're worried about a test. Nobody cares. *burp* Wubba lubba dub dub.

Morty: Rick, are you drunk?
Rick: I'm always drunk, Morty. Sobriety is just, *burp*, the universe's way of making you feel things.

Morty: What's the meaning of life, Rick?
Rick: To make butter pass, Morty. Hah, I'm kidding, there's no meaning. You exist, you suffer, you die. Pass me that bottle.

"""

SPEAKER_USER = "Morty"
SPEAKER_BOT = "Rick"


def build_prompt(history, user_message):
    """history: liste de (user, rick). Retourne le texte à donner à GPT-2."""
    parts = [PERSONA]
    for u, r in history:
        parts.append("%s: %s\n%s: %s\n\n" % (SPEAKER_USER, u, SPEAKER_BOT, r))
    parts.append("%s: %s\n%s:" % (SPEAKER_USER, user_message, SPEAKER_BOT))
    return "".join(parts)


def cut_reply(generated):
    """Ne garde que la réplique de Rick : coupe dès que Morty reprend la parole
    ou qu'une ligne vide apparaît."""
    text = generated.lstrip(" ")
    text = re.split(r"\n\s*\n|\n\s*(?:%s|%s)\s*:" % (SPEAKER_USER, SPEAKER_BOT), text)[0]
    text = text.replace("\n", " ").strip()
    # évite de finir au milieu d'une phrase si on peut couper proprement
    m = list(re.finditer(r"[.!?*](?=\s|$)", text))
    if m and m[-1].end() > len(text) * 0.5:
        text = text[: m[-1].end()]
    return text


def drunkify(text, level=0.5, rng=None):
    """Ajoute rots, bégaiements et Morty-spam. level entre 0 (sobre) et 1 (HS)."""
    rng = rng or random
    level = max(0.0, min(1.0, level))
    words = text.split(" ")
    out = []
    for w in words:
        # bégaiement sur certains mots : "w-w-what"
        if len(w) > 2 and w[0].isalpha() and rng.random() < 0.06 * level * 2:
            w = "%s-%s" % (w[0], w)
        out.append(w)
        # rot en milieu de phrase (si le modèle n'en a pas déjà mis)
        if rng.random() < 0.05 * level * 2 and "*burp*" not in w:
            out.append("*burp*")
    text = " ".join(out)
    # "Morty" en début de réplique, façon Rick
    if "Morty" not in text and rng.random() < level:
        text = "Morty, " + text[0].lower() + text[1:] if text else text
    # petit rot final de temps en temps
    if rng.random() < 0.3 * level and not text.endswith("*burp*"):
        text += " *burp*"
    # pas de rots collés : "*burp* *burp*" -> "*burp*"
    text = re.sub(r"(\*burp\*\s*){2,}", "*burp* ", text).strip()
    return text
