import logging
from typing import Optional

import langdetect
import requests

logger = logging.getLogger("openwpm")

LIBRE_TRANSLATE_HOST = "host.docker.internal"
LIBRE_TRANSLATE_PORT = 5000
LIBRE_TRANSLATE_URL = f"http://{LIBRE_TRANSLATE_HOST}:{LIBRE_TRANSLATE_PORT}"


def detect_language(text: str) -> Optional[str]:
    try:
        return langdetect.detect(text)
    except:
        return None


def translate(
    text: str, source_language: Optional[str] = None, verbose: bool = False
) -> str:
    text = text.lower()
    if source_language is None:
        try:
            source_language = langdetect.detect(text)
        except:
            if verbose:
                logger.info("Failed to detect language. Returning original text.")
            return text

    if source_language == "en":
        if verbose:
            logger.info("Text is in English. No need for translation.")
        return text

    try:
        text = requests.post(
            url=f"{LIBRE_TRANSLATE_URL}/translate",
            data={"q": text, "source": source_language, "target": "en"},
        ).json()["translatedText"]
        if verbose:
            logger.info(f"Translated text from {source_language} to en")
    except Exception as e:
        if verbose:
            logger.info(
                f"Failed to translate text from {source_language} to en. Returning untranslated text: {e}"
            )
    return text


def prepare_for_translation(text: str) -> str:
    return (
        text.lower()
        .replace("afvis ", "afvise ")
        .replace("alle ablehnen", "ablehnen alle")
    )
