import argparse
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def main():
    parser = argparse.ArgumentParser(description="Stream speech synthesis with Chatterbox.")
    parser.add_argument("text", help="Text to synthesize.")
    parser.add_argument("--language_id", "-l", default="en", help="Language code, e.g., 'en', 'fr'.")
    parser.add_argument("--audio_prompt", "-a", help="Path to an audio prompt for voice cloning.")
    parser.add_argument("--output", "-o", default="streaming-output.wav", help="Output WAV file path.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    chunks = []
    for chunk in model.generate_stream(args.text, language_id=args.language_id, audio_prompt_path=args.audio_prompt):
        chunks.append(chunk)

    wav = torch.cat(chunks, dim=1)
    ta.save(args.output, wav, model.sr)

if __name__ == "__main__":
    main()

