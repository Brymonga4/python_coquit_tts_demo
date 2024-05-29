import os
import torch
from services.util import Utilities
from services.examples import Models_Examples


CURRENT_EXECT_PATH = os.getcwd()
TEST_FILES_PATH = os.path.join(CURRENT_EXECT_PATH,"app","src", "samples")
OUTPUT_FILES_PATH = os.path.join(CURRENT_EXECT_PATH,"app","src", "output")

# List available 🐸TTS models
# print(TTS().list_models())

device = "cuda" if torch.cuda.is_available() else "cpu"

text = Utilities.read_text_file(os.path.join(TEST_FILES_PATH, "text", "tag_sample.txt"))


# Text to speech to a file
# ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
""" Models_Examples.xtts_from_sample_audio_and_text_to_output_audio(device,
                                                                text, 
                                                                os.path.join(TEST_FILES_PATH, "donquixote_my_voice_spa.wav"),
                                                                "en",
                                                                os.path.join(OUTPUT_FILES_PATH, "output_xtts_v2.wav")
                                                                )

Models_Examples.your_tts_from_sample_audio_and_text_to_output_audio(device,
                                                                text, 
                                                                os.path.join(TEST_FILES_PATH, "donquixote_my_voice_spa.wav"),
                                                                "en",
                                                                os.path.join(OUTPUT_FILES_PATH, "output_your_tts.wav")
                                                                )

Models_Examples.vctk_from_audio_to_audio_output(device,
                                                os.path.join(TEST_FILES_PATH, "donquixote_my_voice_spa.wav"),
                                                os.path.join(TEST_FILES_PATH, "leonor.wav"),
                                                os.path.join(OUTPUT_FILES_PATH, "output_vctk.wav")
                                                )

Models_Examples.tacotron2_GER_from_text_to_output_audio(device,
                                                    "Ich bin eine Testnachricht.",
                                                    os.path.join(OUTPUT_FILES_PATH, "output_GER.wav")
                                                    ) """

Models_Examples.xtts_from_sample_audio_and_text_to_output_audio(device,
                                                                text, 
                                                                os.path.join(TEST_FILES_PATH, "tag_sample.m4a"),
                                                                "es",
                                                                os.path.join(OUTPUT_FILES_PATH, "output_tag.wav")
                                                                )
