import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from swa.audio_stimulus import AudioStimulus, normalize_audio_provenance


class AudioStimulusTests(unittest.TestCase):
    def test_missing_provenance_never_attributes_text_to_user(self):
        stimulus = AudioStimulus.from_payload({"texto": "frase reconhecida"})
        self.assertEqual(stimulus.path, "unknown")
        self.assertIn("origem desconhecida", stimulus.prompt(""))
        self.assertIn("não validada", stimulus.prompt(""))
        self.assertNotIn("Usuário disse", stimulus.prompt(""))

    def test_synthetic_event_has_explicit_path(self):
        stimulus = AudioStimulus.from_payload({
            "texto": "texto de voz sintética",
            "proveniencia_audio": {
                "propagation_path": "synthetic",
                "source_class": "configured_synthetic",
            },
        })
        self.assertEqual(stimulus.path, "synthetic")
        self.assertIn("áudio sintético declarado", stimulus.evaluation_input())

    def test_unrecognized_path_falls_back_to_unknown(self):
        stimulus = AudioStimulus.from_payload({
            "texto": "texto",
            "proveniencia_audio": {"propagation_path": "owner_confirmed"},
        })
        self.assertEqual(stimulus.path, "unknown")

    def test_payload_cannot_self_assert_verified_speaker(self):
        provenance = normalize_audio_provenance({
            "propagation_path": "user_candidate",
            "speaker_attribution": "verified",
        })
        self.assertEqual(provenance["speaker_attribution"], "unverified")


if __name__ == "__main__":
    unittest.main()
