#!/usr/bin/env python3
"""
Скрипт для создания тестовых аудиофайлов (без внешних зависимостей).

Генерирует WAV-файлы с помощью стандартной библиотеки (wave, struct, math, random):
- test_music.wav  — чистый синус 440 Гц (3 сек)
- test_noise.wav  — белый шум (2 сек)
- test_speech.wav — смесь синуса и шума (3 сек), имитирует речь
"""

import os
import sys
import math
import random
import struct
import wave
from pathlib import Path


SAMPLE_RATE = 44100  # Гц
SAMPLE_WIDTH = 2     # 16 бит
NUM_CHANNELS = 1     # моно


def _clip_int16(value: float) -> int:
    """Обрезает значение в диапазон 16-битового PCM и возвращает int."""
    return max(-32768, min(32767, int(value)))


def _write_wav(path: Path, samples: list[int]) -> None:
    """Записывает массив int16 сэмплов в WAV-файл."""
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        # Упаковываем в little-endian 16-bit signed
        frames = struct.pack('<' + 'h' * len(samples), *samples)
        wf.writeframes(frames)


def generate_sine(duration_s: float, freq_hz: float, amplitude: float = 0.5) -> list[int]:
    """Генерирует синусоидальный сигнал заданной длительности и частоты."""
    num_samples = int(SAMPLE_RATE * duration_s)
    amp = amplitude * 32767.0
    samples = []
    for n in range(num_samples):
        value = amp * math.sin(2.0 * math.pi * freq_hz * (n / SAMPLE_RATE))
        samples.append(_clip_int16(value))
    return samples


def generate_white_noise(duration_s: float, amplitude: float = 0.4) -> list[int]:
    """Генерирует белый шум."""
    num_samples = int(SAMPLE_RATE * duration_s)
    amp = amplitude * 32767.0
    return [_clip_int16(random.uniform(-amp, amp)) for _ in range(num_samples)]


def mix_signals(a: list[int], b: list[int], gain_a: float = 1.0, gain_b: float = 1.0) -> list[int]:
    """Смешивает два сигнала одинаковой длины с заданными коэффициентами усиления."""
    length = min(len(a), len(b))
    mixed = []
    for i in range(length):
        value = a[i] * gain_a + b[i] * gain_b
        mixed.append(_clip_int16(value))
    return mixed


def generate_music_with_rhythm(total_duration_s: float = 4.0, bpm: int = 120) -> list[int]:
    """Генерирует простую музыкальную зарисовку: метроном + мелодия из нескольких нот.

    - Ритм: короткие щелчки (высокочастотные импульсы) на каждую долю
    - Мелодия: последовательность нот A4 (440 Hz), C5 (~523.25 Hz), E5 (~659.25 Hz), D5 (~587.33 Hz)
    """
    beat_len_s = 60.0 / bpm
    num_samples_total = int(SAMPLE_RATE * total_duration_s)
    samples = [0] * num_samples_total

    # 1) Ритм: на каждую долю добавляем короткий щелчок (импульс с экспоненциальным спадом)
    click_len = int(SAMPLE_RATE * 0.02)  # 20 ms
    for beat_idx in range(int(total_duration_s / beat_len_s) + 1):
        start = int(beat_idx * beat_len_s * SAMPLE_RATE)
        for i in range(click_len):
            idx = start + i
            if idx >= num_samples_total:
                break
            # Высокочастотный щелчок (8 кГц) с быстрым спадом
            t = i / SAMPLE_RATE
            val = math.sin(2 * math.pi * 8000 * t) * math.exp(-t * 80)
            samples[idx] = _clip_int16(samples[idx] + int(val * 0.4 * 32767))

    # 2) Мелодия: несколько нот по полудоле
    notes = [440.0, 523.25, 659.25, 587.33]  # A4, C5, E5, D5
    note_duration_s = beat_len_s / 2.0
    pos = 0
    note_idx = 0
    while pos < num_samples_total:
        freq = notes[note_idx % len(notes)]
        dur = min(int(note_duration_s * SAMPLE_RATE), num_samples_total - pos)
        env = _adsr_envelope(dur, a=0.01, d=0.05, s_level=0.85, r=0.05)
        for i in range(dur):
            t = i / SAMPLE_RATE
            val = math.sin(2 * math.pi * freq * t) * env[i]
            samples[pos + i] = _clip_int16(samples[pos + i] + int(val * 0.35 * 32767))
        pos += dur
        note_idx += 1

    return samples

def _adsr_envelope(total_len: int, a: float = 0.03, d: float = 0.05, s_level: float = 0.7, r: float = 0.07) -> list[float]:
    """Генерирует простую ADSR-огибающую длиной total_len сэмплов."""
    attack = int(total_len * a)
    decay = int(total_len * d)
    release = int(total_len * r)
    sustain = total_len - attack - decay - release
    if sustain < 0:
        sustain = 0
    env = []
    # Attack
    for i in range(max(attack, 1)):
        env.append(i / max(attack, 1))
    # Decay
    for i in range(max(decay, 1)):
        t = i / max(decay, 1)
        env.append(1.0 - t * (1.0 - s_level))
    # Sustain
    env.extend([s_level] * max(sustain, 0))
    # Release
    for i in range(max(release, 1)):
        t = 1.0 - (i / max(release, 1))
        env.append(s_level * max(t, 0.0))
    # Нормализация длины
    return env[:total_len]


def _formant_vowel(duration_s: float, base_freq: float, formants: list[float], amplitude: float = 0.4) -> list[int]:
    """Генерирует гласный звук с набором формант (сумма нескольких синусов)."""
    num_samples = int(SAMPLE_RATE * duration_s)
    env = _adsr_envelope(num_samples)
    samples = []
    for n in range(num_samples):
        t = n / SAMPLE_RATE
        # Базовый тон (псевдо-основа речи)
        val = math.sin(2 * math.pi * base_freq * t)
        # Форманты (усиливаем определенные частоты)
        for idx, f in enumerate(formants, start=1):
            val += 0.6 / idx * math.sin(2 * math.pi * f * t)
        samples.append(_clip_int16(val * amplitude * 32767.0 * env[n]))
    return samples


def _consonant_noise(duration_s: float, amplitude: float = 0.25) -> list[int]:
    """Короткая шумовая согласная (например, 'с', 'ш')."""
    num_samples = int(SAMPLE_RATE * duration_s)
    env = _adsr_envelope(num_samples, a=0.01, d=0.02, s_level=0.6, r=0.03)
    amp = amplitude * 32767.0
    return [_clip_int16(random.uniform(-amp, amp) * env[n]) for n in range(num_samples)]


def _concat(parts: list[list[int]]) -> list[int]:
    """Конкатенирует несколько массивов сэмплов."""
    out: list[int] = []
    for p in parts:
        out.extend(p)
    return out


def generate_speech_like_sequence(total_duration_s: float = 3.0) -> list[int]:
    """Генерирует последовательность, похожую на слоги речи (≈ total_duration_s)."""
    # Набор «гласных» с формантами (приближенно):
    vowels = [
        # (название, [форманты в Гц]) — примерные значения
        ("a", [700, 1100, 2450]),
        ("o", [500, 900, 2400]),
        ("e", [530, 1840, 2480]),
        ("i", [270, 2290, 3010]),
        ("u", [300, 870, 2240]),
    ]
    base_freqs = [110.0, 140.0, 180.0]  # псевдо-основной тон речи (разный «голос»)

    # Составим простую фразу из чередования согласных и гласных
    parts: list[list[int]] = []
    target_len = int(SAMPLE_RATE * total_duration_s)
    current_len = 0
    random.seed(42)
    while current_len < target_len:
        # Согласная
        cons = _consonant_noise(duration_s=0.05 + random.random() * 0.05, amplitude=0.2)
        parts.append(cons)
        current_len += len(cons)
        if current_len >= target_len:
            break
        # Гласная
        vowel_name, formants = random.choice(vowels)
        base = random.choice(base_freqs)
        vow = _formant_vowel(duration_s=0.12 + random.random() * 0.18, base_freq=base, formants=formants, amplitude=0.35)
        parts.append(vow)
        current_len += len(vow)

        # Короткая пауза (тишина)
        pause_len = int(SAMPLE_RATE * (0.03 + random.random() * 0.05))
        parts.append([0] * pause_len)
        current_len += pause_len

    # Обрезаем/подгоняем точную длину
    concat = _concat(parts)
    return concat[:target_len]
def create_test_audio_files() -> None:
    """Создает тестовые WAV-аудиофайлы в папке examples/."""
    out_dir = Path('examples')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Музыка: ритм + несколько нот (≈ 4 сек, 120 BPM)
    print("Создаю тестовый музыкальный файл (ритм + несколько нот, ≈4 сек)...")
    music_samples = generate_music_with_rhythm(total_duration_s=4.0, bpm=120)
    _write_wav(out_dir / 'test_music.wav', music_samples)

    # 2) Шум: белый шум, 2 секунды
    print("Создаю тестовый файл с шумом (2 сек)...")
    noise_samples = generate_white_noise(duration_s=2.0, amplitude=0.5)
    _write_wav(out_dir / 'test_noise.wav', noise_samples)

    # 3) «Речь»: синтетические слоги с огибающей и формантами (≈ 3 сек)
    print("Создаю тестовый файл, имитирующий речь (≈3 сек)...")
    speech_like = generate_speech_like_sequence(total_duration_s=3.0)
    _write_wav(out_dir / 'test_speech.wav', speech_like)

    print("✅ Тестовые аудиофайлы созданы в папке examples/:")
    print("   - test_music.wav (музыка)")
    print("   - test_noise.wav (шум)")
    print("   - test_speech.wav (речь)")


def check_existing_files() -> None:
    """Проверяет существующие файлы в папке examples"""
    examples_dir = Path('examples')
    if not examples_dir.exists():
        return
    
    audio_files = []
    for file_path in examples_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {'.mp3', '.wav', '.m4a', '.flac'}:
            audio_files.append(file_path.name)
    
    if audio_files:
        print(f"\n📁 Найдены существующие аудиофайлы в examples/:")
        for file_name in audio_files:
            print(f"   - {file_name}")
        print("   Эти файлы можно использовать для тестирования.")

if __name__ == "__main__":
    print("🔧 Создание тестовых файлов для аудиоанализатора")
    print("=" * 50)
    
    check_existing_files()
    create_test_audio_files()
    
    print("\n✅ Готово! Теперь можно тестировать аудиоанализатор:")
    print("   python main.py examples")
