import unittest
from unittest.mock import MagicMock, patch

from prototype_3_gestures.realtime import build_parser, resolve_camera
from prototype_3_gestures.tts import TTSManager, speak_text


class TestTTSIntegration(unittest.TestCase):
    def test_parser_has_new_tts_and_camera_options(self):
        parser = build_parser()
        args = parser.parse_args(["--tts-engine", "vieneu", "--tts-voice", "Mai Anh", "--camera", "A16"])
        self.assertEqual(args.tts_engine, "vieneu")
        self.assertEqual(args.tts_voice, "Mai Anh")
        self.assertEqual(args.camera, "A16")

    @patch("prototype_3_gestures.realtime.list_available_cameras")
    def test_resolve_camera_by_name(self, mock_list):
        mock_list.return_value = [
            "HD Webcam",
            "A16 của Thọ (Windows Virtual Camera)",
            "OBS Virtual Camera",
        ]
        idx, name = resolve_camera("A16")
        self.assertEqual(idx, 1)
        self.assertIn("A16", name)

        idx_webcam, name_webcam = resolve_camera("webcam")
        self.assertEqual(idx_webcam, 0)
        self.assertEqual(name_webcam, "HD Webcam")

        idx_direct, _ = resolve_camera("1")
        self.assertEqual(idx_direct, 1)

    @patch("prototype_3_gestures.realtime.list_available_cameras")
    def test_resolve_camera_not_found(self, mock_list):
        mock_list.return_value = ["HD Webcam"]
        with self.assertRaises(ValueError):
            resolve_camera("NonExistentCamera")

    def test_tts_manager_disabled(self):
        manager = TTSManager(enabled=False)
        manager.speak("Xin chào")
        self.assertTrue(manager._queue.empty())
        manager.stop()

    def test_tts_manager_synthesize_mock(self):
        manager = TTSManager(engine="mock", enabled=True)
        with patch.object(manager, "_synthesize_and_play") as mock_play:
            manager.speak("Xin chào")
            # Wait for queue processing
            manager._queue.join()
            mock_play.assert_called_with("Xin chào")
        manager.stop()

    def test_resolve_vieneu_voice(self):
        from prototype_3_gestures.tts import resolve_vieneu_voice

        # Exact
        self.assertEqual(resolve_vieneu_voice("Trúc Ly"), "Trúc Ly")
        self.assertEqual(resolve_vieneu_voice("Mai Anh"), "Mai Anh")
        # Case insensitive
        self.assertEqual(resolve_vieneu_voice("trúc ly"), "Trúc Ly")
        self.assertEqual(resolve_vieneu_voice("TRÚC LY"), "Trúc Ly")
        # Unaccented
        self.assertEqual(resolve_vieneu_voice("Truc Ly"), "Trúc Ly")
        self.assertEqual(resolve_vieneu_voice("truc ly"), "Trúc Ly")
        self.assertEqual(resolve_vieneu_voice("mai anh"), "Mai Anh")
        self.assertEqual(resolve_vieneu_voice("minh quan"), "Minh Quân")
        # Default fallback
        self.assertEqual(resolve_vieneu_voice(""), "Trúc Ly")
        self.assertEqual(resolve_vieneu_voice(None), "Trúc Ly")


if __name__ == "__main__":
    unittest.main()
