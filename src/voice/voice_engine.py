import pyttsx3
import time

class VoiceEngine:

    def __init__(self):

        self.engine = pyttsx3.init()

        # ====================================================
        # PROFESSIONAL SETTINGS
        # ====================================================

        self.engine.setProperty('rate', 135)

        self.engine.setProperty('volume', 1.0)

    def speak(self, text):

        self.engine.say(text)

        self.engine.runAndWait()

        time.sleep(1)

    def speak_slowly(self, diagnosis, recommendation):

        intro = (
            "Rhizo Net analysis is complete."
        )

        self.speak(intro)

        self.speak(
            f"Detected condition: {diagnosis}"
        )

        self.speak(
            f"Recommended action: {recommendation}"
        )

        self.speak(
            "Please consult agricultural experts if symptoms continue."
        )