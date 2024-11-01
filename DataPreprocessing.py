from functools import reduce
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Pipeline
import time
import subprocess
import librosa
import noisereduce as nr
import soundfile as sf
import os

from NoiseReduction import noise_reduction


class DataPreprocessor:
    def __init__(self, print_cost=False):
        self.print_cost = print_cost

    def converting_mp3_to_wav(self, input_file, output_file):
        subprocess.run(["ffmpeg", "-i", input_file, output_file])

    def reduce_noise(self, input_audio, output_audio):
        # Load audio file
        audio_data, sr = librosa.load(input_audio, sr=None)

        # Reduce noise
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sr)

        # Save the denoised audio to a file
        sf.write(output_audio, reduced_noise_audio, sr)

    def pipeline_noise_reduction(self, input_folder, output_folder):
        i = 1
        for dirname, _, filenames in os.walk(input_folder):
            for filename in filenames:
                audio_path = os.path.join(dirname, filename)
                print(f"\nprocessing noise {i} audio path {audio_path}")
                noise_reduction(audio_path, output_folder)
                i += 1

    def speaker_diarization(self, audio_file='audio_file.wav'):
        start_time = time.time()
        total_audio_time = AudioSegment.from_wav(audio_file).duration_seconds

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token='your-token')

        if (self.print_cost):
            print(f"Start Diarization with {audio_file}")

        diarization = pipeline(audio_file, max_speakers=2)

        # Initialize counters for total time and speakers' time
        speakers_turn = dict(list())
        speakers_duration = dict()

        for turn, _, speaker in diarization.itertracks(yield_label=True):

            if speaker in speakers_turn:
                speakers_turn[speaker].append((turn.start, turn.end))
            else:
                speakers_turn[speaker] = [(turn.start, turn.end)]

            if self.print_cost:
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker={speaker}")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        #output_folder/cc_enhance/002-0c-en.wav
        speakers_number = len(speakers_turn.keys())
        if speakers_number <= 1:
            list_turns = speakers_turn.values()
            if self.print_cost:
                print(f"There is {speakers_number} speaker.")
            return [(0, total_audio_time)] if len(list_turns) == 1 else list_turns
        else:
            starter_speaker = min(speakers_turn, key=lambda speaker: speakers_turn[speaker][0][0])
            # participant = 'SPEAKER_00' if '_01' in investigator else 'SPEAKER_01'
            duration_calculator = lambda l: reduce(lambda acc, dt: acc + dt,
                                                   map(lambda t: t[1] - t[0], l))
            for key in speakers_turn.keys():
                speakers_duration[key] = duration_calculator(speakers_turn[key])
            participant = max(speakers_duration, key=speakers_duration.get)
            investigator = 'SPEAKER_00' if '_01' in participant else 'SPEAKER_01'
            participant_audio = []
            speaker_and_turn = tuple()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == participant and len(speaker_and_turn) == 0:
                    participant_audio.append((turn.start, turn.end))
                if speaker == participant and len(speaker_and_turn) > 0:
                    if str(speaker_and_turn[0]) == investigator:
                        participant_audio.append((min(speaker_and_turn[1] + 0.1, turn.start) , turn.end))
                    else :
                        t = participant_audio.pop()
                        participant_audio.append((t[0], turn.end))
                speaker_and_turn = (speaker, turn.end)

            if self.print_cost:
                # Print the total audio time and time for each speaker
                print(f"Total audio time: {total_audio_time:.2f} seconds")
                print(f"{'participant' if starter_speaker == participant else 'investigator'} start the conversation")
                print(f"participant total time: {speakers_duration[participant]:.2f} seconds")
                print(f"investigator total time: {speakers_duration[investigator]:.2f} seconds")

                print(f"Patient Segment {speakers_turn[participant]}")
                print(f"Patient Full Segment {participant_audio}")
            return participant_audio

    def extract_participant_audio(self, input_audio, output_audio):
        # Load the audio file using pydub
        original_audio = AudioSegment.from_mp3(input_audio)

        participant_segments = self.speaker_diarization(input_audio)

        participant_audio = AudioSegment.silent(duration=0)

        for start, end in participant_segments:
            start_ms = start * 1000
            end_ms = end * 1000
            participant_audio += original_audio[start_ms:end_ms]

        participant_audio.export(output_audio, format='wav')

    def generate_id(self, id):
        if id > 1000 :
            return str(id)
        elif 100 < id < 999:
            return "0"+str(id)
        elif 10 < id < 99:
            return "00"+str(id)
        else :
            return "000"+str(id)

    def pipeline_investigator_and_Patient_Audio_Separation(self, input_folder, output_folder, i=1):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for dirname, _, filenames in os.walk(input_folder):
            c_folder = dirname.split('/')[-1]
            if not os.path.exists(os.path.join(output_folder, c_folder)) and c_folder != input_folder:
                os.makedirs(os.path.join(output_folder, c_folder))
            for filename in filenames:
                audio_path = os.path.join(dirname, filename)
                unique_filename = self.generate_id(i) + '-' + filename
                output_path = os.path.join(output_folder, c_folder, unique_filename)
                print(f"\nprocessing {i} dirname {dirname}, filename {filename}")
                self.extract_participant_audio(audio_path, output_path)

                i += 1
