import numpy as np
import librosa
import os
import soundfile as sf

def calculate_rmse():
    # Base directories
    generated_dir = os.path.join('samples', 'generated')
    modified_dir = os.path.join('samples', 'modified')

    # Get list of files in each directory
    generated_files = os.listdir(generated_dir)
    modified_files = os.listdir(modified_dir)

    # Iterate over each file in the generated directory
    for gen_file in generated_files:
        # Construct the corresponding file name in the modified directory
        mod_file = gen_file.replace('generated', 'modified')

        # Ensure the corresponding file exists in the modified directory
        if mod_file in modified_files:
            # Construct the full file paths
            gen_path = os.path.join(generated_dir, gen_file)
            mod_path = os.path.join(modified_dir, mod_file)

            # Load the audio files
            y1, sr1 = librosa.load(gen_path)
            y2, sr2 = librosa.load(mod_path)

            # Check if the sampling rates match
            assert sr1 == sr2, f"Sampling rates must be the same for {gen_path} and {mod_path}"

            # If the signals are not of the same length, pad the shorter signal with zeros
            if len(y1) > len(y2):
                y2 = np.pad(y2, (0, len(y1) - len(y2)))
            elif len(y2) > len(y1):
                y1 = np.pad(y1, (0, len(y2) - len(y1)))

            # Calculate the RMSE
            rmse = np.sqrt(np.mean((y1 - y2)**2))

            print(f"RMSE for {gen_file} and {mod_file}: {rmse}")
        else:
            print(f"No corresponding modified file for {gen_file}")

def calculate_corrcoef():
    # directory of generated and modified sounds
    generated_dir = './samples/generated/'
    modified_dir = './samples/modified/'

    # list all files in generated and modified directories
    generated_files = os.listdir(generated_dir)
    modified_files = os.listdir(modified_dir)

    # for each generated file
    for gen_file in generated_files:
        # find the corresponding modified file
        for mod_file in modified_files:
            # compare the prefix of the filenames, excluding the extension and the last part after "_"
            gen_prefix = os.path.splitext(gen_file)[0].rsplit('_', 1)[0]
            mod_prefix = os.path.splitext(mod_file)[0].rsplit('_', 1)[0]
            if gen_prefix == mod_prefix:

                # read the two sound files
                gen_data, _ = sf.read(generated_dir + gen_file)
                mod_data, _ = sf.read(modified_dir + mod_file)
                
                # handle different lengths
                min_len = min(len(gen_data), len(mod_data))
                gen_data = gen_data[:min_len]
                mod_data = mod_data[:min_len]
                
                # compute correlation
                correlation = np.corrcoef(gen_data, mod_data)[0, 1]
                print(f'Correlation between {gen_file} and {mod_file} is {correlation}')

if __name__ == "__main__":
    calculate_corrcoef()