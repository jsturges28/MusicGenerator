from preprocess1 import MinMaxNormaliser
import librosa

class SoundGenerator:
    """SoundGenerator is responsible for generating audios from spectrograms."""

    def __init__(self, model, hop_length):
        self.model = model
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormaliser(0,1)

    def generate_vae(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.model.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations
    
    #UNET has no latent representations in this case

    def generate_unet(self, spectrograms, min_max_values):
        generated_spectrograms = self.model.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []

        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape log spectrogram
            log_spectrogram = spectrogram[:, :, 0]
            # apply denormalization
            denorm_log_spec = self._min_max_normalizer.denormalise(log_spectrogram, min_max_value["min"], min_max_value["max"])
            # log spectrogram -> linear spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # apply Griffin-Lim
            signal = librosa.griffinlim(spec, hop_length=self.hop_length)
            # append signal to "signals"
            signals.append(signal)

        return signals