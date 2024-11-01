import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf
import os
from itertools import groupby
from PIL import Image
import matplotlib.cm as cm

class FeatureEngineering:
    def __init__(self, print_cost=False):
        self.print_cost = print_cost

    def split_audio(self, audio_path, output_folder, num_chunks):
        audio = AudioSegment.from_file(audio_path)

        chunk_length = len(audio) / num_chunks

        # Extraire le nom du fichier et le chemin
        audio_filename = os.path.basename(audio_path)
        base_name, _ = os.path.splitext(audio_filename)

        # Extraire le nom du dossier et le premier élément
        folder_name = os.path.dirname(audio_path).split('/')[-1]  # Nom du dossier
        id_audio = base_name.split('-')[0]  # Premier élément avant le tiret
        rest_of_name = '-'.join(base_name.split('-')[1:])  # Le reste du nom

        # Créer le chemin de sortie dynamique
        output_subfolder = os.path.join(output_folder, folder_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # Diviser l'audio et enregistrer les chunks
        for i in range(num_chunks):
            start_time = i * chunk_length
            end_time = start_time + chunk_length if i < num_chunks - 1 else len(audio)
            chunk = audio[start_time:end_time]
            chunk_name = f"{id_audio}-{i+1}-{rest_of_name}.wav"
            chunk_path = os.path.join(output_subfolder, chunk_name)
            chunk.export(chunk_path, format="wav")

    def pipeline_dividing_audio_into_chunks(self, input_folder, output_folder, num_chunks):
        i = 1
        for dirname, _, filenames in os.walk(input_folder):
            for filename in filenames:
                audio_path = os.path.join(dirname, filename)
                print(f"\nprocessing {i} audio path {audio_path}")
                self.split_audio(audio_path, output_folder, num_chunks)
                i += 1

    def normalise_chunk_amplitude(self, chunk_path, output_folder):
        if not os.path.isfile(chunk_path):
            raise ValueError(f"The provided path '{chunk_path}' is not a valid file.")

        try:
            y, sr = librosa.load(chunk_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file '{chunk_path}': {e}")

        y_normalized = y / np.max(np.abs(y))

        audio_filename = os.path.basename(chunk_path)
        folder_name = os.path.dirname(chunk_path).split('/')[-1]
        output_subfolder = os.path.join(output_folder, folder_name)
        normalised_chunk_path = os.path.join(output_subfolder, audio_filename)
        os.makedirs(output_subfolder, exist_ok=True)

        try:
            sf.write(normalised_chunk_path, y_normalized, sr)
        except Exception as e:
            raise RuntimeError(f"Failed to write audio file to '{normalised_chunk_path}': {e}")
        #return y

    def pipeline_normalise_chunk_amplitude(self, input_folder, output_folder):
        i = 1
        for dirname, _, filenames in os.walk(input_folder):
            for filename in filenames:
                audio_path = os.path.join(dirname, filename)
                print(f"\nnormalising amplitude {i} chunk path {audio_path}")
                self.normalise_chunk_amplitude(audio_path, output_folder)
                i += 1

    def load_chunk(self, file_path):
        # Load chunk file
        audio = AudioSegment.from_file(file_path)
        return audio

    def find_voice_start_end(self, file_path, silence_threshold=-50.0):
        """
        Finds the start and end positions of the voice in an audio file.
        :param file_path: Path to the audio file.
        :param silence_threshold: Threshold in dBFS below which audio is considered silent.
        :return: Tuple containing the start and end positions of the voice (in milliseconds).
        """
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        voice_start = None
        voice_end = None
        audio_length = len(audio)

        # find the start and end of the voice
        for i in range(audio_length):
            if audio[i].dBFS > silence_threshold:
                if voice_start is None:
                    voice_start = i
                voice_end = i

        # No silence at the beginning
        if voice_start is None:
            voice_start = 0

        # verifier No silence at the end
        if voice_end is None:
            voice_end = audio_length

        return voice_start , voice_end   # Convert to seconds

    def normalise_chunk_length(self, chunk_path, output_folder, length):
        # load the chunk of data
        chunk = self.load_chunk(chunk_path)
        # check if the chunk is longer than the length
        if len(chunk) > length:
            start_ms, end_ms = self.find_voice_start_end(chunk_path)
            if self.print_cost:
                print(f"chunk is longer {len(chunk)} than the length {length}")
                print(f"start time {start_ms} ms")
                print(f"end time {end_ms} ms")

            # Calculate the length of the voice segment
            voice_length = end_ms - start_ms

            if voice_length > length:
                if self.print_cost:
                    print(f"voice segment is longer {voice_length} than the target length {length}")
                # Normalize by truncating the voice segment to the target length
                chunk = chunk[start_ms:start_ms + length]
            else:

                silence = AudioSegment.silent(duration=(length - voice_length)/2)
                chunk = silence + chunk[start_ms:end_ms] + silence
        else:
            # If the chunk is shorter than the target length, pad with silence
            chunk = chunk + AudioSegment.silent(duration=length - len(chunk))

        if self.print_cost:
            print("length of the chunk", len(chunk))

        audio_filename = os.path.basename(chunk_path)
        folder_name = os.path.dirname(chunk_path).split('/')[-1]
        output_subfolder = os.path.join(output_folder, folder_name)
        os.makedirs(output_subfolder, exist_ok=True)
        output_chunk_path = os.path.join(output_subfolder, audio_filename)
        chunk.export(output_chunk_path, format="wav")

    def pipeline_normalise_chunk_length(self, input_folder, output_folder, chunk_length):
        i = 1
        for dirname, _, filenames in os.walk(input_folder):
            for filename in filenames:
                audio_path = os.path.join(dirname, filename)
                print(f"\nnormalising length {i} chunk path {audio_path}")
                self.normalise_chunk_length(audio_path, output_folder, chunk_length)
                i += 1

    def group_chunks_by_audio_id(self, list_paths):
        path_to_keypath = lambda chunk_path: (os.path.basename(chunk_path).split('-')[0], chunk_path)
        data = list(map(path_to_keypath, list_paths))
        data.sort(key=lambda x: x[0])
        # Group by key
        grouped_data = {key: list(map(lambda x: x[1], group)) for key, group in groupby(data, key=lambda x: x[0])}
        return grouped_data

    def extract_mfcc_features(self, audio_file):
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)

        # Set parameters
        frame_length = int(0.025 * sr)  # 0.025 seconds
        hop_length = int(0.01 * sr)  # 0.01 seconds

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)  # First derivatives

        # Combine MFCC and its first derivative
        mfcc_combined = np.vstack((mfcc, mfcc_delta))

        # Normalize and convert to RGB
        normalized_mfcc = (mfcc_combined - np.min(mfcc_combined)) / (np.max(mfcc_combined) - np.min(mfcc_combined))
        colormap = cm.get_cmap('viridis')
        mfcc_rgb = colormap(normalized_mfcc)[:, :, :3]  # Convert to RGB

        # Scale to 8-bit
        return (mfcc_rgb * 255).astype(np.uint8)

    def combine_mfcc_images(self, chunks_path):
        # Get MFCC RGB arrays for each chunk
        image_arrays = [self.extract_mfcc_features(chunk_path) for chunk_path in chunks_path]

        # Stack arrays vertically
        combined_image_array = np.vstack(image_arrays)

        return combined_image_array

    def pipeline_combine_mfcc_images(self, input_folder):
        cc_data_paths, cd_data_paths = [], []

        # Collect paths for 'cc' and 'cd' labels
        for dirname, _, filenames in os.walk(input_folder):
            for filename in filenames:
                audio_path = os.path.join(dirname, filename)
                if 'cc' in dirname.split('/')[-1]:
                    cc_data_paths.append(audio_path)
                else:
                    cd_data_paths.append(audio_path)

        # Group audio chunks by IDs for 'cc' and 'cd'
        cc_chunks_group = self.group_chunks_by_audio_id(cc_data_paths).values()
        cd_chunks_group = self.group_chunks_by_audio_id(cd_data_paths).values()

        # Combine MFCC images for each group of chunks
        cc_data = [self.combine_mfcc_images(chunks_path) for chunks_path in cc_chunks_group]
        cd_data = [self.combine_mfcc_images(chunks_path) for chunks_path in cd_chunks_group]

        # Stack data and create target labels
        all_data = np.array(cc_data + cd_data)
        target = np.array([0] * len(cc_data) + [1] * len(cd_data))
        print(all_data.shape)
        return all_data, target