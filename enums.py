#
#
#   Enums
#
#

import enum


class Language(enum.Enum):
    SPANISH = "SPANISH"
    ENGLISH = "ENGLISH"

    @property
    def azure_language(self):
        if self == Language.SPANISH:
            return "es-ES"
        elif self == Language.ENGLISH:
            return "en-US"
        else:
            raise NotImplementedError()

    @property
    def azure_voice(self):
        if self == Language.SPANISH:
            return "es-ES-EstrellaNeural"
        elif self == Language.ENGLISH:
            return "en-US-JaneNeural"
        else:
            raise NotImplementedError()

    @property
    def nltk_language(self):
        if self == Language.SPANISH:
            return "spanish"
        elif self == Language.ENGLISH:
            return "english"
        else:
            raise NotImplementedError()
