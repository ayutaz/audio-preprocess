from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from loguru import logger
import traceback
import os

def prevent_clipping(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val > threshold:
        audio = audio * (threshold / max_val)
    return audio

def loudness_norm(
    audio: np.ndarray, rate: int, peak: float = -1.0, loudness: float = -23.0, block_size: float = 0.400
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        audio: audio data
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)

    Returns:
        loudness normalized audio
    """

    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    loudness_pre = meter.integrated_loudness(audio)

    audio = pyln.normalize.loudness(audio, loudness_pre, loudness)
    audio = pyln.normalize.peak(audio, peak)

    return audio

def loudness_norm_file(input_file: Union[str, Path], output_file: Union[str, Path], peak: float = -1.0, loudness: float = -23.0, min_block_size: float = 0.1) -> Tuple[str, bool]:
    try:
        audio, rate = sf.read(str(input_file))

        # 最小ブロックサイズの計算（ファイルの長さの10%か0.1秒のいずれか大きい方）
        min_samples = max(int(min_block_size * rate), int(len(audio) * 0.1))
        block_size = min(0.400, len(audio) / rate)  # 0.4秒かファイルの長さの短い方

        if len(audio) < min_samples:
            logger.warning(f"Audio file {input_file} is too short (length: {len(audio)}, minimum: {min_samples}). Skipping.")
            return str(input_file), False  # 処理をスキップしたことを示す

        audio = loudness_norm(audio, rate, peak, loudness, block_size)

        audio = prevent_clipping(audio)

        chunk_size = 100000
        with sf.SoundFile(output_file, 'w', samplerate=rate,
                        channels=audio.shape[1] if len(audio.shape) > 1 else 1) as f:
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                f.write(chunk)

        return str(input_file), True  # 処理が成功したことを示す

    except Exception as e:
        logger.error(f"Error in loudness_norm_file for {input_file}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return str(input_file), False  # エラーが発生したことを示す