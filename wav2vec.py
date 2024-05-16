import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 음성 데이터 로드
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

# WAV2Vec2 모델 및 전처리기 로드
def load_model_and_processor(model_name):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return model, processor

# 음성 데이터 전처리
def preprocess_audio(waveform, sample_rate, processor):
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    return inputs.input_values, inputs.input_length

# 추론 함수 정의
def inference(model, inputs):
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription

# 메인 함수
def main(file_path, model_name):
    # 음성 데이터 로드
    waveform, sample_rate = load_audio(file_path)

    # WAV2Vec2 모델 및 전처리기 로드
    model, processor = load_model_and_processor(model_name)

    # 음성 데이터 전처리
    inputs = preprocess_audio(waveform, sample_rate, processor)

    # 추론
    transcription = inference(model, inputs)

    return transcription

if __name__ == "__main__":
    file_path = "your_audio_file.wav"
    model_name = "facebook/wav2vec2-large-960h"
    result = main(file_path, model_name)
    print("Transcription:", result)
