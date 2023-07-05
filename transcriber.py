import argparse
import openai
import os
from datetime import datetime
from pydub import AudioSegment

"""
Usage: `pipenv run python transcriber.py [-h] [--input_filename INPUT_FILENAME] [--skip [SKIP_SECONDS]]`
"""

class Transcriber:
    def __init__(self, input_filename: str, skip_seconds: float):
        self.skip_seconds = skip_seconds
        self.input_file = os.path.join("input", input_filename)
        self.segment_duration_minutes = 5
        self.segment_output_format = "mp3"
        self.segment_filename_template = "segment_{0}.{1}"
        self.results_dir = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _split_file(self) -> int:
        audio = AudioSegment.from_wav(self.input_file)[self.skip_seconds*1000:]

        # Split the file into segments and export them
        for i, segment in enumerate(audio[::self.segment_duration_minutes*60*1000]):
            output_filename = os.path.join(self.results_dir, self.segment_filename_template.format(i, self.segment_output_format))
            segment.export(output_filename, format=self.segment_output_format)
        
        return i

    def _transcribe(self, count_segments: int) -> None:
        for i in range(0, count_segments):
            segment_audio_filename = os.path.join(self.results_dir, self.segment_filename_template.format(i, self.segment_output_format))
            segment_audio= open(segment_audio_filename, "rb")
            transcript = openai.Audio.transcribe("whisper-1", segment_audio, response_format="verbose_json", language="en")

            segment_json_filename = os.path.join(self.results_dir, self.segment_filename_template.format(i, 'json'))
            with open(segment_json_filename, "w", encoding="utf-8") as json_file:
                json_file.write(str(transcript))

            segment_txt_filename = os.path.join(self.results_dir, self.segment_filename_template.format(i, 'txt'))
            with open(segment_txt_filename, "w", encoding="utf-8") as text_file:
                for i, segment in enumerate(transcript['segments']):
                    text_file.write(segment['text'] + '\n')

    def main(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        print("Splitting...")
        count_segments = self._split_file()
        print("Splitting complete!")
        print("Transcribing...")
        self._transcribe(count_segments)
        print("Transcription complete!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog=f"pipenv run python {os.path.basename(__file__)}",
        description="Transcribe an audio file using Whisper API",
    )

    ap.add_argument(
        "--input_filename",
        dest="input_filename",
        default="input.wav",
        help="The input filename",
        type=str,
        required=False,
    )

    ap.add_argument(
        "--skip",
        dest="skip_seconds",
        default=1.0,
        nargs="?",
        help="Duration to skip from the beginning of the input file in number of seconds",
        type=float,
        required=False,
    )

    args = ap.parse_args()

    transcriber = Transcriber(
        input_filename=args.input_filename, skip_seconds=args.skip_seconds
    )
    transcriber.main()