import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
import cog
from cog import types

class Predictor(cog.BasePredictor):
    def setup(self):
        self.model = MusicGen.get_pretrained('melody')
        self.model.set_generation_params(duration=8)  # generate 8 seconds.
    
    def predict(self,
                audio: types.File = cog.Input(description="Input audio file"),
                description: str = cog.Input(description="Music description")
    ) -> Path:
        """Run a single prediction on the model"""
        melody, sr = torchaudio.load(audio)
        wav = self.model.generate_with_chroma([description], melody.unsqueeze(0), sr)

        # Will save under {description}.wav, with loudness normalization at -14 db LUFS.
        output_file = f'{description}.wav'
        audio_write(output_file, wav[0].cpu(), self.model.sample_rate, strategy="loudness")
        
        return Path(output_file)
