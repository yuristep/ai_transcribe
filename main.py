#!/usr/bin/env python3
"""
Аудиоанализатор с использованием OpenAI API.

Поддерживает распознавание речи и классификацию аудиофайлов.
Использует GPT-4o-transcribe для высококачественного распознавания речи
с приоритетом русского языка и альтернативным английским языком.
"""

import argparse
import logging
import os
import sys
import tempfile
import warnings
from datetime import datetime
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import openai
from dotenv import load_dotenv

# Подавляем предупреждения FFmpeg и других библиотек
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*FFmpeg.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*demucs.*")

# Опциональные зависимости
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Отключаем загрузку расширений torio FFmpeg, чтобы избежать ошибок libtorio_ffmpeg*.pyd
# Используем системный ffmpeg и подавляем расширения torio FFmpeg
os.environ["TORIO_DISABLE_EXTENSIONS"] = "1"
os.environ["TORIO_LOG_LEVEL"] = "ERROR"

# Загружаем переменные окружения
load_dotenv()

# =============================================================================
# Единые значения по умолчанию и промты
# =============================================================================

# Языковые настройки
DEFAULT_PRIMARY_LANGUAGE = "ru"
DEFAULT_SECONDARY_LANGUAGE = "en"

# Настройки аудио
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac']
DEFAULT_FOLDER = "examples"

# Настройки Demucs (оптимизированы для лучшего качества разделения вокала)
DEMUCS_MODEL = "htdemucs"  # Доступные модели: mdx, htdemucs, htdemucs_ft, mdx_q, mdx_extra_q
DEMUCS_SHIFTS = 4  # Больше сдвигов -> чище маски (1-10, больше = лучше качество, но медленнее)
DEMUCS_OVERLAP = 0.4  # Большее перекрытие улучшает качество (0.1-0.5)
DEMUCS_DEVICE = "cuda"  # Устройство: "cpu" или "cuda"
DEMUCS_JOBS = 4  # Количество потоков (1-4)
DEMUCS_TWO_STEMS = "vocals"  # Извлекать только вокал для ускорения

# Настройки librosa
VOCAL_FREQ_MIN = 80
VOCAL_FREQ_MAX = 8000
VOCAL_FREQ_PARTIAL_MIN = 8000
VOCAL_FREQ_PARTIAL_MAX = 16000
VOCAL_MASK_FULL = 1.0
VOCAL_MASK_PARTIAL = 0.5

# Пороги для транскрипции
MIN_RESULT_LENGTH_FOR_ALT_LANG = 10
MIN_RESULT_LENGTH_FOR_MUSIC_PROMPT = 100
MIN_TEXT_LENGTH_FOR_POST_PROCESS = 10

# Настройки оптимизации транскрипции
MAX_TRANSCRIPTION_ATTEMPTS = 3  # Максимальное количество попыток транскрипции
SKIP_MUSIC_PROMPT_IF_SHORT = True  # Пропускать музыкальный промпт для коротких результатов

# Настройки LLM
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.1

# Настройки классификации
CLASSIFICATION_MAX_TOKENS = 200
CLASSIFICATION_TEMPERATURE = 0.1

# Настройки логирования
LOG_DIR = "log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_ENABLE_LOGGING = True

# Настройки пост-обработки по умолчанию
DEFAULT_ENABLE_POST_PROCESS = True

# Промпты для транскрипции
TRANSCRIPTION_PROMPT = """Дай дословную ПОЛНУЮ транскрипцию без перевода и перефразирования. Сохраняй язык оригинала, не смешивай языки. Не интерпретируй смысл, не сокращай, не исправляй. Если часть неразборчива — оставь как есть или пропусти без домыслов. ВНИМАНИЕ: Если это музыка с повторяющимся припевом - распознай ВСЕ повторения. Если это плач ребенка или нечеловеческие звуки - НЕ придумывай слова, верни пустую строку.

Орфография и пунктуация: используй корректную русскую орфографию и пунктуацию (точки, запятые, вопросительные знаки). Сохраняй разговорные формы (например: «Чё») как в оригинале, но избегай искажённых слов (например «Посони» → «Позвони», если по звучанию это очевидно)."""

MUSIC_TRANSCRIPTION_PROMPT = """Это музыкальная композиция с вокалом. Распознай ВСЕ слова песни от начала до конца. Дай дословную ПОЛНУЮ транскрипцию без перевода и перефразирования. Сохраняй язык оригинала, не интерпретируй, не сокращай. Внимательно слушай начало песни - там тоже есть слова. Если есть повторяющийся припев - распознай ВСЕ повторения полностью. Если это инструментальная музыка без слов - верни пустую строку. ВАЖНО: Распознай весь текст песни, включая все куплеты и припевы. Не останавливайся на первом фрагменте - слушай до конца.

Орфография и пунктуация: ставь знаки препинания, оформляй предложения, сохраняй разговорные формы («Чё») и правильные слова («Позвони», «Не говори нет»), избегай искажений (не «Посони»)."""

# Промпт для пост-обработки
POST_PROCESS_PROMPT = """Ты — эксперт по коррекции текстов песен и транскрипций.
Твоя задача — исправить ошибки в транскрипции песни, сохранив оригинальный смысл и стиль.

Правила коррекции:
1. Исправляй грамматические ошибки
2. Восстанавливай недостающие слова по контексту
3. Сохраняй ритм и стиль песни
4. Не меняй смысл и эмоциональную окраску
5. Сохраняй повторения (припевы)
6. Исправляй только очевидные ошибки
7. Сохраняй разговорные формы и просторечие как в оригинале (например: «Чё»)
8. Предпочитай устойчивые выражения и идиомы (например: «не говори нет» вместо «не говори мне», если контекст песни допускает это лучше)
9. Оформляй предложения корректной пунктуацией: точки, вопросы и запятые. Вопросительные интонации заканчивай знаком «?»
10. Избегай ложных отрицаний: не меняй «мне выносишь» на «не выносишь», если смыслу и ритму больше соответствует дательный падеж «мне».

Отвечай только исправленным текстом, без объяснений."""

# Промпт для классификации аудио
CLASSIFICATION_PROMPT = """Ты — специализированная система для анализа аудиозаписей.
Твоя единственная задача — классифицировать входной аудиофайл в одну из категорий: музыка, речь, шум.

ВАЖНО: Ты анализируешь РЕЗУЛЬТАТ ТРАНСКРИПЦИИ аудиофайла, а не сам аудиофайл. Это означает, что:
- Если в аудио была музыка с вокалом, то в транскрипции будут слова песни
- Если в аудио была речь, то в транскрипции будет связный текст
- Если в аудио был шум, то в транскрипции будет мало слов или галлюцинации

Категории классификации:

Музыка (music):
- Это песня, музыкальная композиция с вокалом
- Характерные признаки в транскрипции:
  * Повторяющиеся фразы или слова (припев)
  * Рифмы и ритмические паттерны
  * Музыкальные звуки: "ла-ла-ла", "на-на-на", "о-о-о", "а-а-а"
  * Эмоциональные восклицания: "Ауф!", "Ой!", "Ах!"
  * Короткие фразы без логической связи
  * Слова, которые звучат как песня, а не как обычная речь
  * Повторяющиеся слова или фразы (припев песни)
  * Ритмические паттерны в тексте

Речь (speech):
- Это обычная речь, разговор, диалог
- Характерные признаки в транскрипции:
  * Связные предложения с логической структурой
  * Обычные слова и фразы
  * Знаки препинания: точки, запятые, восклицательные знаки
  * Логическая последовательность мыслей
  * Отсутствие повторяющихся музыкальных паттернов
  * Естественная речь без ритма

Шум (noise):
- Это неречевые звуки, шум, техника, плач ребенка
- Характерные признаки в транскрипции:
  * Очень короткие или бессмысленные фрагменты
  * Отсутствие связной речи
  * Случайные слова или звуки
  * Описания звуков: "[Music]", "[звук]", "[шум]"
  * Повторяющиеся гласные: "ааа", "ууу", "ооо" (детский плач)
  * Повторяющиеся согласные: "шшш", "щщщ", "ссс", "ззз", "hhh"
  * Бессмысленные фразы: "Всем новичкам, всем новичкам"
  * Короткие бессвязные слова

Примеры типовых шумов (не музыка и не речь):
- Городской фон: движение транспорта, гул улицы, толпа, шаги
- Звуки машин: двигатель, сигнал, шелест шин, торможение
- Вода: дождь, шум волн, водопад, шипение воды
- Ветер: завывание, свист ветра, порывы
- Бытовые/механические: кондиционер, вентилятор, скрип дверей, стук, аплодисменты

Правила классификации:
1. Если транскрипция содержит связную речь с логической структурой → speech
2. Если транскрипция содержит повторяющиеся фразы, рифмы, музыкальные звуки → music
3. Если транскрипция короткая, бессвязная или содержит описания звуков → noise
4. При сомнениях между music и speech: если есть повторения или музыкальные звуки → music
5. Детский плач (ааа, ууу, ооо) → noise
6. Белый шум с бессмысленными словами → noise

ВАЖНЫЕ ПРИМЕРЫ:
- "Скажи мне, не говори мне, что ты гонишь" → music (песня с повторениями)
- "Привет, как дела?" → speech (обычная речь)
- "Ааааааа" → noise (детский плач)
- "Всем новичкам, всем новичкам" → noise (белый шум)
- "Find me, find me, You can find" → music (песня с припевом)

Формат ответа:
Отвечай только в формате JSON, без лишнего текста:
{
  "classification": "music | speech | noise",
  "confidence": 0.0-1.0,
  "reasoning": "Краткое объяснение выбора: какие признаки были решающими"
}

Ограничения:
- Не добавляй ничего, кроме JSON
- Не предлагай альтернативных категорий
- Всегда возвращай одно значение в поле "classification"
- Уровень уверенности (confidence) должен отражать баланс признаков:
  * 0.9–1.0 → очень высокая уверенность
  * 0.6–0.89 → умеренная уверенность
  * <0.6 → низкая уверенность, признаки неоднозначны"""


# Настройки производительности
ENABLE_AUDIO_ENHANCEMENT = True  # Отключаем улучшение аудио по умолчанию для скорости
SKIP_DEMUCS_ON_ERROR = False  # Пропускать Demucs при ошибках

# Вспомогательный контекстный менеджер для глушения шума сторонних библиотек (torio/ffmpeg/torchaudio/demucs)
@contextmanager
def suppress_external_noise():
    previous_disable_level = logging.root.manager.disable
    try:
        # Понижаем уровень логирования глобально
        logging.disable(logging.CRITICAL)

        # Настраиваем переменные окружения для снижения болтливости расширений
        os.environ.setdefault("TORIO_LOG_LEVEL", "ERROR")
        os.environ.setdefault("TORIO_DISABLE_EXTENSIONS", "1")  # игнорировать попытки загрузки ffmpeg расширений
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # иногда шумит при импортe demucs

        # Отводим stdout/stderr во внутренние буферы на время вызова Demucs
        fake_out, fake_err = io.StringIO(), io.StringIO()
        with redirect_stdout(fake_out), redirect_stderr(fake_err):
            yield
    finally:
        # Восстанавливаем уровень логирования
        logging.disable(previous_disable_level)


class AudioAnalyzer:
    """
    Анализатор аудиофайлов с использованием OpenAI API.
    
    Поддерживает распознавание речи и классификацию аудио на русском и английском языках.
    """
    
    def _get_demucs_params(self) -> dict:
        """
        Возвращает параметры Demucs из блока настроек.
        
        Args:
            None
            
        Returns:
            Словарь с параметрами Demucs
        """
        return {
            'model': DEMUCS_MODEL,
            'shifts': DEMUCS_SHIFTS,
            'overlap': DEMUCS_OVERLAP,
            'jobs': DEMUCS_JOBS,
        }
    
    def __init__(self, primary_language: str = DEFAULT_PRIMARY_LANGUAGE, secondary_language: str = DEFAULT_SECONDARY_LANGUAGE, enable_post_process: bool = DEFAULT_ENABLE_POST_PROCESS, enable_logging: bool = DEFAULT_ENABLE_LOGGING) -> None:
        """
        Инициализация анализатора аудио.
        
        Args:
            primary_language: Основной язык распознавания (по умолчанию: ru)
            secondary_language: Альтернативный язык распознавания (по умолчанию: en)
            enable_post_process: Включить пост-обработку с LLM (по умолчанию: False)
            enable_logging: Включить логирование в файл (по умолчанию: True)
            
        Raises:
            ValueError: Если OPENAI_API_KEY не найден в переменных окружения
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не найден в переменных окружения")
        
        openai.api_key = self.api_key
        self.primary_language = primary_language
        self.secondary_language = secondary_language
        self.enable_post_process = enable_post_process
        self.enable_logging = enable_logging
        self._region_error_notified = False  # чтобы не дублировать сообщения об ограничении региона
        
        # Настройка логирования (если включено)
        if self.enable_logging:
            self._setup_logging()
        else:
            # Отключаем логирование и создаем заглушку
            logging.getLogger().setLevel(logging.CRITICAL)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
    
    def _setup_logging(self) -> None:
        """
        Настраивает логирование для отладки.
        """
        # Создаем папку log если не существует
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(exist_ok=True)
        
        # Создаем имя файла лога с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"session_{timestamp}.log"
        
        # Настраиваем логирование
        logging.basicConfig(
            level=logging.DEBUG,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Логирование настроено. Файл лога: {log_file}")
    
    def _polish_punctuation_ru(self, text: str) -> str:
        """
        Лёгкая нормализация русской пунктуации и идиом после пост-обработки.
        Не меняет смысл, лишь расставляет границы предложений и устойчивые выражения.
        """
        try:
            import re
            s = text.strip()
            # Устойчивое выражение
            s = re.sub(r"\bне\s+говори\s+мне\b", "не говори нет", s, flags=re.IGNORECASE)
            # Разделяем длинные фразы по характерным кускам
            s = re.sub(r",\s*(ч[ёе]\s+ты\s+гонишь)", r". \1", s, flags=re.IGNORECASE)
            s = re.sub(r",\s*(снова\s+мозг\s+мне\s+выносишь)", r". \1", s, flags=re.IGNORECASE)
            # Вопросительная интонация
            s = re.sub(r"\b(ч[ёе]\s+ты\s+гонишь)\.?", r"\1?", s, flags=re.IGNORECASE)
            # Точка после повелительного в начале
            s = re.sub(r"^(Позвони\s+мне)(,|$)", r"\1.", s, flags=re.IGNORECASE)
            # Свести множественные пробелы
            s = re.sub(r"\s+", " ", s)
            # Нормализуем пробелы перед пунктуацией
            s = re.sub(r"\s+([,.!?])", r"\1", s)
            return s.strip()
        except Exception:
            return text

    def _post_process_transcription(self, text: str, debug: bool = False) -> str:
        """
        Пост-обработка транскрипции с помощью LLM для коррекции ошибок.
        
        Args:
            text: Исходный текст транскрипции
            debug: Включить отладочный вывод
            
        Returns:
            Исправленный текст
        """
        if not text or len(text.strip()) < MIN_TEXT_LENGTH_FOR_POST_PROCESS:
            return text
        
        self.logger.debug(f"ПОСТ-ОБРАБОТКА: Исходный текст: '{text}'")
        
        try:
            system_prompt = POST_PROCESS_PROMPT

            response = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Исправь ошибки в транскрипции песни:\n\n{text}"}
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE
            )
            
            corrected_text = response.choices[0].message.content.strip()
            self.logger.debug(f"ПОСТ-ОБРАБОТКА: Исправленный текст: '{corrected_text}'")
            
            # Лёгкая доводка пунктуации/идиом для русского
            if self.primary_language == "ru":
                corrected_text = self._polish_punctuation_ru(corrected_text)
            
            return corrected_text
            
        except Exception as e:
            self.logger.warning(f"ПОСТ-ОБРАБОТКА: Ошибка коррекции: {e}")
            return text
    
    def _enhance_audio_for_transcription(self, file_path: str) -> str:
        """
        Улучшает качество аудио для лучшей транскрипции.
        Использует Demucs для разделения вокала и фоновой музыки.
        При недоступности Demucs использует librosa для базового выделения вокала.
        
        Args:
            file_path: Путь к исходному аудиофайлу
            
        Returns:
            Путь к улучшенному файлу (или исходному, если улучшение невозможно)
        """
        # Если улучшение аудио отключено, возвращаем исходный файл
        if not ENABLE_AUDIO_ENHANCEMENT:
            self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Отключено для ускорения работы")
            return file_path
        
        # Предобработка 1: лёгкая очистка входного сигнала для лучшей сегрегации (не меняем исходный файл)
        preclean_file = self._preclean_for_demucs(file_path)

        # Получаем параметры Demucs из блока настроек
        demucs_params = self._get_demucs_params()
        self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Параметры Demucs: {demucs_params}")
        self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Попытка улучшения файла (после предочистки): {preclean_file}")
            
        try:
            # Пытаемся использовать Demucs для качественного разделения вокала
            try:
                self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Импортируем Demucs...")
                from demucs import separate
                import torch
                import tempfile
                import os
                from pathlib import Path
                
                self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Demucs успешно импортирован")
                self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Используем Demucs для разделения вокала: {file_path}")
                
                # Создаем временную папку для результатов Demucs
                temp_dir = Path(tempfile.mkdtemp())
                
                # Используем Demucs для разделения аудио с параметрами качества
                # Разделяем на: vocals, drums, bass, other
                separate.main([
                    str(preclean_file),
                    "--out", str(temp_dir),
                    "-n", demucs_params['model'],
                    "-d", (DEMUCS_DEVICE if DEMUCS_DEVICE in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")),
                    "--shifts", str(demucs_params['shifts']),
                    "--overlap", str(demucs_params['overlap']),
                    "--two-stems", DEMUCS_TWO_STEMS,
                    "--float32",
                    "-j", str(demucs_params['jobs'])
                ])
                
                # Ищем файл с вокалом
                # Demucs создает структуру: temp_dir/model_name/track_name/vocals.{wav|mp3}
                model_name = demucs_params['model']
                track_name = Path(file_path).stem
                vocals_file_wav = temp_dir / model_name / track_name / "vocals.wav"
                vocals_file_mp3 = temp_dir / model_name / track_name / "vocals.mp3"
                vocals_file = vocals_file_wav if vocals_file_wav.exists() else vocals_file_mp3
                # Если по ожидаемому пути нет, ищем глубоко любой vocals.*
                if not vocals_file.exists():
                    import glob
                    candidates = glob.glob(str(temp_dir / "**" / "vocals.*"), recursive=True)
                    if candidates:
                        vocals_file = Path(candidates[0])
                
                if vocals_file.exists():
                    # Создаем финальный файл с вокалом рядом с исходным: {name}_vocals.{ext}
                    orig_path = Path(file_path)
                    # Сохраняем "как есть" без принудительной конвертации
                    final_vocals_path = orig_path.with_name(f"{orig_path.stem}_vocals{vocals_file.suffix}")
                    final_vocals_file = str(final_vocals_path)
                    import shutil
                    # Перезаписываем, если файл уже существует
                    try:
                        dest_path = Path(final_vocals_file)
                        if dest_path.exists():
                            dest_path.unlink()
                    except Exception:
                        pass
                    try:
                        shutil.move(str(vocals_file), final_vocals_file)
                    except Exception:
                        shutil.copy2(vocals_file, final_vocals_file)

                    # Двухшаговая доочистка вокала: вычитание no_vocals и лёгкий EQ
                    try:
                        self._refine_vocals_with_background(final_vocals_file, temp_dir, model_name, track_name)
                    except Exception:
                        pass

                    # Дополнительная очистка вокала от остаточного аккомпанемента (без смены формата)
                    try:
                        self._cleanup_vocals_file(final_vocals_file)
                    except Exception:
                        pass
                    
                    # Очищаем временную папку
                    shutil.rmtree(temp_dir)
                    
                    self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Создан файл с выделенным вокалом: {final_vocals_file}")
                    return final_vocals_file
                else:
                    self.logger.warning(f"УЛУЧШЕНИЕ АУДИО: Файл с вокалом не найден: {vocals_file}")
                    # Очищаем временную папку
                    shutil.rmtree(temp_dir)
                    return file_path
                
            except ImportError as e:
                self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Demucs недоступен: {e}")
                # Переходим к альтернативному методу с librosa
                return self._enhance_with_librosa(file_path)
            except Exception as e:
                self.logger.warning(f"УЛУЧШЕНИЕ АУДИО: Ошибка обработки с Demucs: {e}")
                if SKIP_DEMUCS_ON_ERROR:
                    self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Пропускаем Demucs из-за ошибки")
                    return file_path
                # Переходим к альтернативному методу с librosa
                return self._enhance_with_librosa(file_path)
            
        except Exception as e:
            self.logger.warning(f"УЛУЧШЕНИЕ АУДИО: Общая ошибка обработки: {e}")
            return file_path

    def _preclean_for_demucs(self, file_path: str) -> str:
        """
        Предочистка исходного файла для лучшего разделения Demucs: HPF ~120 Гц и мягкий high-shelf >10 кГц.
        Возвращает путь к временно сохранённому WAV с предобработкой.
        """
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            p = Path(file_path)
            y, sr = librosa.load(str(p), sr=None, mono=True)
            if y is None or len(y) == 0:
                return file_path
            # HPF
            sos = librosa.iirfilter(2, 120.0, btype='highpass', ftype='butter', fs=sr, output='sos')
            y = librosa.sosfilt(sos, y)
            # лёгкий high-shelf >10 кГц
            S = librosa.stft(y)
            freqs = librosa.fft_frequencies(sr=sr)
            shelf = np.ones_like(freqs)
            shelf[freqs >= 10000] = 10 ** (-2/20)  # -2 дБ
            y = librosa.istft(S * shelf[:, None])
            # временный файл рядом с исходным
            tmp = p.with_name(f"{p.stem}_preclean.wav")
            sf.write(str(tmp), y, sr)
            return str(tmp)
        except Exception:
            return file_path
    
    def _enhance_with_librosa(self, file_path: str) -> str:
        """
        Альтернативный метод улучшения аудио с помощью librosa.
        
        Args:
            file_path: Путь к исходному аудиофайлу
            
        Returns:
            Путь к улучшенному файлу (или исходному, если улучшение невозможно)
        """
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            from pathlib import Path
            
            self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Используем librosa для выделения вокала: {file_path}")
            
            # Загружаем аудио
            y, sr = librosa.load(file_path, sr=None, mono=True)
            
            # Простое выделение вокала с помощью спектрального вычитания
            # Вычисляем спектрограмму
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Создаем маску для вокала (частоты примерно 80-8000 Гц)
            freqs = librosa.fft_frequencies(sr=sr)
            vocal_mask = np.zeros_like(magnitude)
            
            # Усиливаем частоты вокального диапазона
            for i, freq in enumerate(freqs):
                if VOCAL_FREQ_MIN <= freq <= VOCAL_FREQ_MAX:  # Вокальный диапазон
                    vocal_mask[i, :] = VOCAL_MASK_FULL
                elif VOCAL_FREQ_PARTIAL_MIN < freq <= VOCAL_FREQ_PARTIAL_MAX:  # Частично вокальный диапазон
                    vocal_mask[i, :] = VOCAL_MASK_PARTIAL
            
            # Применяем маску
            enhanced_magnitude = magnitude * vocal_mask
            
            # Восстанавливаем аудио
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_y = librosa.istft(enhanced_stft)
            
            # Создаем временный файл
            # Сохраняем строго как {stem}_vocals.wav рядом с исходником
            p = Path(file_path)
            temp_file = str(p.with_name(f"{p.stem}_vocals.wav"))
            sf.write(temp_file, enhanced_y, sr)
            
            self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: Создан файл с выделенным вокалом: {temp_file}")
            return temp_file
            
        except ImportError as e:
            self.logger.debug(f"УЛУЧШЕНИЕ АУДИО: librosa недоступна: {e}")
            return file_path
        except Exception as e:
            self.logger.warning(f"УЛУЧШЕНИЕ АУДИО: Ошибка обработки с librosa: {e}")
            return file_path

    def _cleanup_vocals_file(self, file_path: str) -> None:
        """
        Лёгкая дополнительная очистка вокала от остаточного аккомпанемента:
        - подавление низких частот (< 120 Гц)
        - мягкий high-shelf на верхах, чтобы убрать шипение от перкуссии
        Без изменения формата файла.
        """
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            y, sr = librosa.load(file_path, sr=None, mono=True)
            if y is None or len(y) == 0:
                return
            # Высокочастотная фильтрация: HPF 2-го порядка ~120 Гц
            sos = librosa.iirfilter(2, 120.0, rs=None, btype='highpass', ftype='butter', fs=sr, output='sos')
            y = librosa.sosfilt(sos, y)
            # Лёгкий high-shelf: ослабим диапазон > 10 кГц на -3 дБ эквивалентно (грубая аппроксимация через EQ в частотной области)
            S = librosa.stft(y)
            freqs = librosa.fft_frequencies(sr=sr)
            shelf = np.ones_like(freqs)
            shelf[freqs >= 10000] = 10 ** (-3/20)
            y = librosa.istft(S * shelf[:, None])
            sf.write(file_path, y, sr)
        except Exception:
            return

    def _refine_vocals_with_background(self, vocals_path: str, temp_dir: Path, model_name: str, track_name: str, bg_weight: float = 0.2) -> None:
        """
        Уменьшает остатки аккомпанемента в вокале:
        1) читает no_vocals из временного каталога Demucs
        2) делает мягкое вычитание: vocals = vocals - bg_weight * no_vocals (подгон по длине)
        3) применяет лёгкий EQ (тот же, что в _cleanup_vocals_file)
        Результат перезаписывает в vocals_path (без смены формата/расширения).
        """
        try:
            import soundfile as sf
            import numpy as np
            import librosa
            # Путь к no_vocals
            bg_wav = temp_dir / model_name / track_name / "no_vocals.wav"
            bg_mp3 = temp_dir / model_name / track_name / "no_vocals.mp3"
            bg_file = bg_wav if bg_wav.exists() else (bg_mp3 if bg_mp3.exists() else None)
            if bg_file is None:
                return
            # Чтение
            y_v, sr = librosa.load(vocals_path, sr=None, mono=True)
            y_bg, sr_bg = librosa.load(str(bg_file), sr=sr, mono=True)
            if len(y_bg) != len(y_v):
                m = min(len(y_bg), len(y_v))
                y_bg = y_bg[:m]
                y_v = y_v[:m]
            # Мягкое вычитание
            y = y_v - float(bg_weight) * y_bg
            # Лёгкая очистка (тот же фильтр, что в _cleanup_vocals_file)
            sos = librosa.iirfilter(2, 120.0, btype='highpass', ftype='butter', fs=sr, output='sos')
            y = librosa.sosfilt(sos, y)
            S = librosa.stft(y)
            freqs = librosa.fft_frequencies(sr=sr)
            shelf = np.ones_like(freqs)
            shelf[freqs >= 10000] = 10 ** (-2/20)
            y = librosa.istft(S * shelf[:, None])
            sf.write(vocals_path, y, sr)
        except Exception:
            return
    
    def list_audio_files(self, folder: str) -> List[Path]:
        """
        Находит все аудиофайлы в указанной папке.
        
        Args:
            folder: Путь к папке для поиска
            
        Returns:
            Список путей к найденным аудиофайлам
            
        Raises:
            FileNotFoundError: Если папка не существует
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder}")
        
        audio_files = []
        for file_path in folder_path.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
                continue
            # Исключаем сгенерированные стемы вокала из первичной обработки
            # чтобы избежать зацикливания пайплайна: *_vocals.*
            try:
                if file_path.stem.lower().endswith("_vocals"):
                    continue
            except Exception:
                pass
            audio_files.append(file_path)
        
        return sorted(audio_files)
    
    def choose_file(self, files: List[Path]) -> Optional[Path]:
        """
        Интерактивный выбор файла из списка.
        
        Args:
            files: Список путей к файлам
            
        Returns:
            Выбранный файл или None если отменено
        """
        if not files:
            print("❌ Аудиофайлы не найдены")
            return None
        
        print(f"\n📁 Найденные аудиофайлы:")
        for i, file_path in enumerate(files, 1):
            file_size = file_path.stat().st_size / (1024 * 1024)
            print(f"  {i}. {file_path.name} ({file_size:.1f} МБ)")
        
        print(f"  {len(files) + 1}. ❌ Отмена")
        
        while True:
            try:
                choice = input(f"\nВведите номер файла (1-{len(files)}) или {len(files) + 1} для отмены: ")
                choice_num = int(choice)
                
                if choice_num == len(files) + 1:
                    return None
                elif 1 <= choice_num <= len(files):
                    return files[choice_num - 1]
                else:
                    print(f"❌ Введите число от 1 до {len(files) + 1}")
            except ValueError:
                print("❌ Введите корректное число")
            except KeyboardInterrupt:
                return None
    
    def convert_audio_format(self, file_path: Path) -> str:
        """
        Конвертирует аудиофайл в MP3 формат если необходимо.
        
        Args:
            file_path: Путь к исходному файлу
            
        Returns:
            Путь к конвертированному файлу или исходному если конвертация недоступна
        """
        if AudioSegment is None:
            return str(file_path)
        
        try:
            # Загружаем аудиофайл
            audio = AudioSegment.from_file(str(file_path))
            
            # Конвертируем в MP3
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                audio.export(temp_file.name, format='mp3')
                return temp_file.name
        except Exception as e:
            print(f"Ошибка конвертации аудио: {e}")
            return str(file_path)
    
    def _get_transcription_prompt(self) -> str:
        """
        Возвращает стандартный промпт для транскрипции.
        
        Returns:
            Строка с промптом для транскрипции
        """
        return TRANSCRIPTION_PROMPT
    
    def _get_music_transcription_prompt(self) -> str:
        """
        Возвращает промпт для транскрипции музыкальных композиций.
        
        Returns:
            Строка с промптом для музыкальной транскрипции
        """
        return MUSIC_TRANSCRIPTION_PROMPT
    
    def transcribe_audio(self, file_path: str, debug: bool = False) -> str:
        """
        Распознает речь в аудиофайле с использованием многоуровневого подхода.
        
        Args:
            file_path: Путь к аудиофайлу
            debug: Включить отладочный вывод
            
        Returns:
            Распознанный текст
        """
        self.logger.info(f"ТРАНСКРИПЦИЯ: Начало анализа файла: {file_path}")
        try:
            # Улучшаем качество аудио для лучшей транскрипции
            enhanced_file = self._enhance_audio_for_transcription(file_path)
            
            strict_prompt = self._get_transcription_prompt()
            self.logger.debug(f"ТРАНСКРИПЦИЯ: Использован промпт: {strict_prompt}")
            
            # Попытка 1: Основной язык (ru) на улучшенном файле
            self.logger.debug(f"ТРАНСКРИПЦИЯ: Попытка 1 - основной язык: {self.primary_language}")
            result = self._transcribe_with_language(enhanced_file, self.primary_language, strict_prompt, debug)
            self.logger.debug(f"ТРАНСКРИПЦИЯ: Результат 1: '{result}' (длина: {len(result)})")

            # Попытка 2: Авто-язык (если коротко) — также на улучшенном файле
            if len(result) < MIN_RESULT_LENGTH_FOR_ALT_LANG:
                self.logger.debug("ТРАНСКРИПЦИЯ: Попытка 2 - авто-язык")
                result_auto = self._transcribe_with_language(enhanced_file, None, strict_prompt, debug)
                self.logger.debug(f"ТРАНСКРИПЦИЯ: Результат 2: '{result_auto}' (длина: {len(result_auto)})")
                if len(result_auto) > len(result):
                    result = result_auto

            # Попытка 3: Музыкальный промпт (если всё ещё коротко) — только если по аудиопризнакам это похоже на музыку
            if len(result) < MIN_RESULT_LENGTH_FOR_MUSIC_PROMPT:
                try:
                    looks_musical = self._is_instrumental_music(file_path)
                except Exception:
                    looks_musical = False
                if looks_musical:
                    self.logger.debug("ТРАНСКРИПЦИЯ: Попытка 3 - музыкальный промпт (обнаружены музыкальные признаки)")
                    music_prompt = self._get_music_transcription_prompt()
                    result_music = self._transcribe_with_language(enhanced_file, None, music_prompt, debug)
                    self.logger.debug(f"ТРАНСКРИПЦИЯ: Результат музыкального промпта: '{result_music}' (длина: {len(result_music)})")
                    if len(result_music) > len(result):
                        result = result_music
                    
                    # Английский язык для музыкальных файлов (если всё ещё коротко)
                    if len(result) < MIN_RESULT_LENGTH_FOR_MUSIC_PROMPT:
                        self.logger.debug("ТРАНСКРИПЦИЯ: Доп. шаг - английский язык с музыкальным промптом")
                        result_en_music = self._transcribe_with_language(enhanced_file, "en", music_prompt, debug)
                        self.logger.debug(f"ТРАНСКРИПЦИЯ: Результат англ. музыкального промпта: '{result_en_music}' (длина: {len(result_en_music)})")
                        if len(result_en_music) > len(result):
                            result = result_en_music
                else:
                    self.logger.debug("ТРАНСКРИПЦИЯ: Музыкальные признаки не обнаружены — пропускаем музыкальный промпт")
            
            # Проверяем, что результат не является промптом
            result = self._filter_prompt_from_result(result, strict_prompt)
            
            # Пост-обработка с LLM для коррекции ошибок (если включена)
            if self.enable_post_process and result and len(result.strip()) > MIN_TEXT_LENGTH_FOR_POST_PROCESS:
                self.logger.debug("ТРАНСКРИПЦИЯ: Применяем пост-обработку")
                result = self._post_process_transcription(result, debug)
            
            return result
            
        except Exception as e:
            print(f"Ошибка распознавания речи: {e}")
            return ""
    
    
    
    def _is_likely_hallucination(self, text: str) -> bool:
        """
        Определяет, является ли текст вероятной галлюцинацией модели.
        Фильтруем только очевидные галлюцинации, все остальное доверяем ИИ.
        
        Args:
            text: Текст для анализа
            
        Returns:
            True если текст похож на очевидную галлюцинацию
        """
        if not text or len(text) < 3:
            return False
        
        import re
        
        # 1. Очевидные галлюцинации - очень короткие тексты с символами
        if len(text) <= 5 and re.match(r'^[.!?,#\-_]+$', text):
            self.logger.debug(f"ГАЛЛЮЦИНАЦИЯ: текст '{text}' отфильтрован - только символы")
            return True
        
        # 2. Тексты состоящие только из символов (без букв)
        if re.match(r'^[^а-яёa-z\s]+$', text):
            self.logger.debug(f"ГАЛЛЮЦИНАЦИЯ: текст '{text}' отфильтрован - только символы")
            return True
        
        # 3. Тексты в квадратных скобках (описания звуков)
        if text.startswith('[') and text.endswith(']'):
            self.logger.debug(f"ГАЛЛЮЦИНАЦИЯ: текст '{text}' отфильтрован - описание звука")
            return True
        
        # 4. Тексты с множественными символами подряд (4+ одинаковых символа)
        if re.search(r'(.)\1{3,}', text):  # 4+ одинаковых символа подряд
            self.logger.debug(f"ГАЛЛЮЦИНАЦИЯ: текст '{text}' отфильтрован - множественные символы")
            return True
        
        # 5. Детский плач - повторяющиеся гласные (5+ подряд)
        if re.search(r'[аоуэыиеёюя]{5,}', text.lower()):
            self.logger.debug(f"ГАЛЛЮЦИНАЦИЯ: текст '{text}' отфильтрован - детский плач")
            return True
        
        # 6. Очень короткие тексты (до 3 символов) с повторяющимися словами
        words = text.split()
        if len(text) <= 3 and len(words) <= 1 and len(set(words)) < len(words):
            self.logger.debug(f"ГАЛЛЮЦИНАЦИЯ: текст '{text}' отфильтрован - очень короткие повторения")
            return True
        
        # Все остальное доверяем ИИ для классификации
        return False
    
    def _filter_prompt_from_result(self, result: str, prompt: str) -> str:
        """
        Фильтрует промпт из результата транскрипции и определяет галлюцинации.
        Включает детальное логирование для отладки.
        
        Args:
            result: Результат транскрипции
            prompt: Использованный промпт
            
        Returns:
            Очищенный результат
        """
        self.logger.debug(f"ФИЛЬТРАЦИЯ: Входной текст: '{result}'")
        self.logger.debug(f"ФИЛЬТРАЦИЯ: Промпт: '{prompt}'")
        
        if not result or not prompt:
            self.logger.debug("ФИЛЬТРАЦИЯ: Пустой результат или промпт")
            return result
        
        result_lower = result.lower().strip()
        prompt_lower = prompt.lower().strip()
        
        # 1. Проверяем, содержит ли результат промпт (только если результат очень похож на промпт)
        if result_lower == prompt_lower:
            self.logger.warning(f"ФИЛЬТРАЦИЯ: Результат идентичен промпту - отфильтровано")
            return ""
        
        # Проверяем, содержит ли результат большую часть промпта
        if len(result_lower) > 100 and len(prompt_lower) / len(result_lower) > 0.5:
            # Дополнительная проверка: действительно ли это промпт?
            if prompt_lower[:50] in result_lower or result_lower[:50] in prompt_lower:
                self.logger.warning(f"ФИЛЬТРАЦИЯ: Результат содержит большую часть промпта - отфильтровано")
                return ""
        
        # 2. Проверяем на частичное совпадение с ключевыми фразами промпта
        prompt_keywords = [
            "дословную", "полную", "транскрипцию", "без перевода", 
            "сохраняй язык", "не интерпретируй", "не сокращай", "не исправляй"
        ]
        
        keyword_matches = sum(1 for keyword in prompt_keywords if keyword in result_lower)
        if keyword_matches > len(prompt_keywords) // 2:
            self.logger.warning(f"ФИЛЬТРАЦИЯ: Найдено {keyword_matches} ключевых слов промпта - отфильтровано")
            return ""
        
        # 3. Проверяем на галлюцинации - подозрительные паттерны
        if self._is_likely_hallucination(result_lower):
            self.logger.warning(f"ФИЛЬТРАЦИЯ: Текст определен как галлюцинация - отфильтровано")
            return ""
        
        self.logger.debug(f"ФИЛЬТРАЦИЯ: Текст прошел фильтрацию: '{result}'")
        return result
    
    def _transcribe_with_language(self, file_path: str, language: Optional[str], prompt: str, debug: bool) -> str:
        """
        Выполняет транскрипцию с указанным языком и промптом.
        
        Args:
            file_path: Путь к аудиофайлу
            language: Язык для распознавания (None для автоопределения)
            prompt: Промпт для транскрипции
            debug: Включить отладочный вывод
            
        Returns:
            Распознанный текст
        """
        try:
            with open(file_path, 'rb') as audio_file:
                params = {
                    "model": "gpt-4o-transcribe",
                    "file": audio_file,
                    "response_format": "text",
                    "prompt": prompt
                }
                if language:
                    params["language"] = language
                
                transcript = openai.audio.transcriptions.create(**params)
            
            result = transcript.strip()
            if debug:
                lang_label = language or "автоопределение"
                # Показываем отфильтрованный результат в отладке
                filtered_result = self._filter_prompt_from_result(result, prompt)
                if filtered_result != result:
                    print(f"📝 С языком ({lang_label}): '{result}' → отфильтровано → '{filtered_result}'")
                else:
                    print(f"📝 С языком ({lang_label}): '{result}'")
            
            return result
            
        except Exception as e:
            # Улучшим читаемость типичных сетевых/региональных ошибок
            error_text = str(e)
            if "unsupported_country_region_territory" in error_text or "403" in error_text:
                if debug and not self._region_error_notified:
                    print("❌ Доступ к API ограничен по региону (403). Проверьте VPN/прокси или регион аккаунта OpenAI.")
                    self._region_error_notified = True
            if debug:
                lang_label = language or "автоопределение"
                print(f"❌ Ошибка с языком ({lang_label}): {e}")
            return ""
    
    def classify_audio(self, file_path: str, transcript_text: Optional[str] = None) -> str:
        """
        Классифицирует аудиофайл с использованием ИИ на основе транскрипции.
        
        Args:
            file_path: Путь к аудиофайлу
            transcript_text: Уже полученная транскрипция для этого файла (если есть)
            
        Returns:
            Категория аудио: 'музыка', 'речь' или 'шум'
        """
        try:
            # Если транскрипция уже получена на предыдущем этапе — используем её напрямую,
            # чтобы не выполнять повторную транскрипцию и сохранить согласованность.
            if transcript_text is None:
                strict_prompt = self._get_transcription_prompt()
                transcript_text = self._transcribe_with_language(file_path, self.primary_language, strict_prompt, False)
                transcript_text = self._filter_prompt_from_result(transcript_text, strict_prompt)

            # Эвристика: явный шум по тексту → шум (перекрывает вокализации)
            if transcript_text and self._looks_like_noise(transcript_text):
                return "шум"

            # Эвристика: короткие вокализации («la», «na», «ah», «о-о-о») → музыка
            if transcript_text and self._looks_like_vocalizations(transcript_text):
                return "музыка"
            
            if not transcript_text:
                # Пустая транскрипция: проверим простые аудиопризнаки (ритм/тональность) для инструментальной музыки
                if self._is_instrumental_music(file_path):
                    return "музыка"
                return "шум"
            
            # Используем ИИ для классификации
            classification_result = self._classify_with_ai(transcript_text)

            # Если ИИ дал "шум", но по признакам файл инструментальная музыка — корректируем на "музыка"
            if classification_result == "шум":
                try:
                    if self._is_instrumental_music(file_path):
                        return "музыка"
                except Exception:
                    pass
            
            return classification_result
            
        except Exception as e:
            print(f"Ошибка классификации аудио: {e}")
            return "шум"

    def _looks_like_vocalizations(self, text: str) -> bool:
        """
        Возвращает True, если текст похож на вокализации/междометия без связной речи
        (la-la, na-na, ah, oh, ooo и т.п.), характерные для музыки.
        """
        if not text:
            return False
        import re
        t = text.lower()
        # Наличие повторяющихся вокальных слогов
        patterns = [r"\b(la)+\b", r"\b(na)+\b", r"\b(da)+\b", r"\b(ba)+\b",
                    r"\b(ah|aah|aaa)+\b", r"\b(oh|ooh|ooo)+\b", r"\b(yeah|yea)+\b"]
        if any(re.search(p, t) for p in patterns):
            return True
        # Короткий текст с повторяющимися слогами/междометиями
        words = re.findall(r"[a-zа-яё]+", t)
        if len(" ".join(words)) <= 40 and len(set(words)) <= max(1, len(words)//2):
            return True
        return False

    def _looks_like_noise(self, text: str) -> bool:
        """
        Возвращает True, если текст похож на шум/неречевые звуки (напр. "шшш", "щщщ", "--").
        Используется для явного отнесения к шуму до обращения к ИИ.
        """
        if not text:
            return False
        import re
        t = text.strip().lower()
        # Очень короткая строка без пробелов и пунктуации → шум
        if len(t) <= 3 and ' ' not in t and re.fullmatch(r"[\W_\-–—~]+|[a-zа-яё]+", t):
            # Если это не явная вокализация
            if not self._looks_like_vocalizations(t):
                return True
        # Повторяющиеся согласные, характерные для "шумов" (ш/щ/с/з/h)
        if re.fullmatch(r"[шщсзh]+", t) and len(t) <= 10:
            return True
        # Почти нет слов: 1 короткое "слово" до 3-4 символов
        words = re.findall(r"[a-zа-яё]+", t)
        if len(words) <= 1 and (len(t) <= 4 or (words and len(words[0]) <= 3)):
            if not self._looks_like_vocalizations(t):
                return True
        return False

    def _is_instrumental_music(self, file_path: str) -> bool:
        """
        Грубая проверка на инструментальную музыку по аудиопризнакам, если речи нет.
        Использует librosa: HPSS (гармоническая составляющая), темп/ритм, спектральная плоскостность,
        хрома (гармоничность) и активность онсетов. Возвращает True при преобладании музыкальных признаков.
        """
        try:
            import librosa
            import numpy as np
            y, sr = librosa.load(file_path, sr=None, mono=True)
            if y is None or len(y) == 0:
                return False

            # HPSS: доля гармонической энергии
            H, P = librosa.effects.hpss(librosa.stft(y))
            harm_energy = float(np.sum(np.abs(H)))
            perc_energy = float(np.sum(np.abs(P)))
            total_energy = harm_energy + perc_energy + 1e-9
            harmonic_ratio = harm_energy / total_energy

            # Онсет-энергия и темп
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            onset_mean = float(np.mean(onset_env))

            # Если темп не определился, оценим ритмичность по автокорреляции онсетов
            def has_periodic_onsets(env: np.ndarray) -> bool:
                if env.size < 64:
                    return False
                env = (env - env.mean()) / (env.std() + 1e-9)
                corr = np.correlate(env, env, mode='full')[env.size-1:]
                corr[0] = 0.0
                peak = float(np.max(corr[: min(len(corr), 512)]))
                return peak > 20.0

            # Спектральная плоскостность (музыка обычно менее «плоская», чем шум)
            S_mag, _ = librosa.magphase(librosa.stft(y))
            flatness = float(np.mean(librosa.feature.spectral_flatness(S=S_mag)))

            # Хрома: стабильно выраженная гармоничность (вариативность и энергия)
            chroma = librosa.feature.chroma_stft(S=S_mag, sr=sr)
            chroma_energy = float(np.mean(chroma))
            chroma_var = float(np.mean(np.var(chroma, axis=1)))

            # Признаки (пороги ещё более мягкие для электронной/поп музыки)
            rhythmic = (tempo >= 30.0) or has_periodic_onsets(onset_env)
            harmonic = (harmonic_ratio >= 0.20) or (chroma_energy >= 0.10 and chroma_var >= 0.002)
            non_flat = flatness < 0.70
            onset_active = onset_mean > 0.06

            score = sum([rhythmic, harmonic, non_flat, onset_active])
            return score >= 2
        except Exception:
            return False
    
    def _classify_with_ai(self, transcript_text: str) -> str:
        """
        Классифицирует аудио с использованием ИИ на основе транскрипции.
        
        Args:
            transcript_text: Распознанный текст
            
        Returns:
            Категория аудио: 'музыка', 'речь' или 'шум'
        """
        try:
            system_prompt = CLASSIFICATION_PROMPT

            response = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Проанализируй следующий транскрипт аудио и классифицируй его:\n\n{transcript_text}"}
                ],
                temperature=CLASSIFICATION_TEMPERATURE,
                max_tokens=CLASSIFICATION_MAX_TOKENS
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Парсим JSON ответ
            import json
            try:
                result = json.loads(result_text)
                classification = result.get("classification", "noise")
                
                # Преобразуем английские категории в русские
                if classification == "music":
                    return "музыка"
                elif classification == "speech":
                    return "речь"
                else:
                    return "шум"
                    
            except json.JSONDecodeError:
                # Если JSON не парсится, попробуем извлечь категорию из текста
                result_lower = result_text.lower()
                if "music" in result_lower or "музыка" in result_lower:
                    return "музыка"
                elif "speech" in result_lower or "речь" in result_lower:
                    return "речь"
                else:
                    return "шум"
                    
        except Exception as e:
            print(f"Ошибка ИИ-классификации: {e}")
            return "шум"
    
    def analyze_audio(self, file_path: Path, debug: bool = False) -> Dict[str, Union[str, float, datetime]]:
        """
        Выполняет полный анализ аудиофайла.
        
        Args:
            file_path: Путь к аудиофайлу
            debug: Включить отладочный вывод
            
        Returns:
            Словарь с результатами анализа
        """
        print(f"\n🔍 Анализирую файл: {file_path.name}")
        
        # Конвертируем формат если необходимо
        converted_path = self.convert_audio_format(file_path)
        
        if debug:
            print(f"📁 Полный путь: {file_path.absolute()}")
            print(f"📊 Размер файла: {file_path.stat().st_size / (1024 * 1024):.2f} МБ")
        
        # Распознаем речь
        print("📝 Распознаю речь...")
        transcript = self.transcribe_audio(converted_path, debug)
        
        # Классифицируем аудио (передаем уже полученную транскрипцию, чтобы не запускать STT повторно)
        print("🏷️ Классифицирую аудио...")
        audio_type = self.classify_audio(converted_path, transcript_text=transcript)
        
        if debug:
            print(f"🏷️ Результат классификации: {audio_type}")
        
        # Очищаем временный файл если он был создан
        if converted_path != str(file_path) and os.path.exists(converted_path):
            try:
                os.unlink(converted_path)
            except OSError:
                pass  # Игнорируем ошибки удаления временного файла
        
        return {
            'file_name': file_path.name,
            'file_path': str(file_path.absolute()),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'audio_type': audio_type,
            'transcript': transcript,
            'analysis_time': datetime.now()
        }
    
    def analyze_all_files(self, folder: str, debug: bool = False) -> List[Dict[str, Union[str, float, datetime]]]:
        """
        Выполняет пакетный анализ всех аудиофайлов в папке.
        
        Args:
            folder: Путь к папке с аудиофайлами
            debug: Включить отладочный вывод
            
        Returns:
            Список результатов анализа
        """
        print(f"🔍 Пакетный анализ файлов в папке: {folder}")
        
        try:
            files = self.list_audio_files(folder)
            print(f"✅ Найдено {len(files)} аудиофайлов")
            
            results = []
            for i, file_path in enumerate(files, 1):
                print(f"\n📁 Обрабатываю файл {i}/{len(files)}: {file_path.name}")
                
                result = self.analyze_audio(file_path, debug)
                results.append(result)
                
                print(f"✅ Завершено: {result['audio_type']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка при пакетном анализе: {e}")
            return []
    
    def save_results(self, results: List[Dict[str, Union[str, float, datetime]]], filename: Optional[str] = None, single_file: bool = False) -> str:
        """
        Сохраняет результаты анализа в файл.
        
        Args:
            results: Список результатов анализа
            filename: Имя файла (если не указано, генерируется автоматически)
            single_file: Если True, использует имя файла для одного результата
            
        Returns:
            Путь к сохраненному файлу
        """
        if filename is None:
            if single_file and len(results) == 1:
                # Для одного файла используем имя файла
                file_name = results[0].get('file_name', 'unknown')
                safe_name = Path(file_name).stem.replace(' ', '_').replace('(', '').replace(')', '')
                filename = f"result_{safe_name}.txt"
            else:
                # Для пакетной обработки используем временную метку
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"result_{timestamp}.txt"
        
        filepath = Path(filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("РЕЗУЛЬТАТЫ АНАЛИЗА АУДИОФАЙЛОВ\n")
            f.write("="*60 + "\n")
            f.write(f"Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего файлов: {len(results)}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write("="*40 + "\n")
                f.write(f"ФАЙЛ {i}: {result['file_name']}\n")
                f.write("="*40 + "\n")
                f.write(f"📁 Файл: {result['file_name']}\n")
                f.write(f"📊 Размер: {result['file_size_mb']:.2f} МБ\n")
                f.write(f"🏷️ Тип аудио: {result['audio_type']}\n")
                f.write(f"📝 Распознанный текст:\n")
                f.write(f"   {result['transcript']}\n")
                f.write(f"⏰ Время анализа: {result['analysis_time'].isoformat()}\n\n")
        
        return str(filepath.absolute())


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Создает парсер аргументов командной строки.
    
    Returns:
        Настроенный парсер аргументов
    """
    parser = argparse.ArgumentParser(
        description="Анализатор аудиофайлов с использованием OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
               Примеры использования:
         python main.py examples                    # Интерактивный выбор файла
         python main.py --all-files examples       # Пакетный анализ всех файлов
         python main.py --file examples/song.mp3   # Анализ конкретного файла
         python main.py --all-files examples --debug --save  # С отладкой и сохранением
         python main.py --file examples/song.mp3 --post-process --no-logging  # С пост-обработкой без логирования
         python main.py --file examples/song.mp3 --enable-post-process  # С включенной пост-обработкой по умолчанию
         python main.py --file examples/song.mp3 --fast-mode --no-logging  # Быстрый режим без логирования
         # Параметры Demucs задаются только в блоке настроек в начале файла
        """
    )
    
    parser.add_argument(
        'folder',
        nargs='?',
        default=DEFAULT_FOLDER,
        help=f'Путь к папке с аудиофайлами (по умолчанию: {DEFAULT_FOLDER})'
    )
    
    parser.add_argument(
        '--all-files',
        action='store_true',
        help='Пакетное распознавание для всех аудиофайлов в папке'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Путь к конкретному файлу для анализа (например: examples/song.mp3)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Сохранить результаты работы в файл result_{time}.txt (по умолчанию включено)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Не сохранять результаты в файл'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Подробный вывод параметров, запроса и ответа'
    )
    
    parser.add_argument(
        '--primary-lang',
        type=str,
        default=DEFAULT_PRIMARY_LANGUAGE,
        help=f'Основной язык распознавания (по умолчанию: {DEFAULT_PRIMARY_LANGUAGE})'
    )
    
    parser.add_argument(
        '--secondary-lang',
        type=str,
        default=DEFAULT_SECONDARY_LANGUAGE,
        help=f'Альтернативный язык распознавания (по умолчанию: {DEFAULT_SECONDARY_LANGUAGE})'
    )
    
    parser.add_argument(
        '--post-process',
        action='store_true',
        default=False,
        help='Включить пост-обработку с LLM для коррекции ошибок транскрипции (по умолчанию выключено)'
    )
    
    parser.add_argument(
        '--no-logging',
        action='store_true',
        default=False,
        help='Отключить логирование в файл (по умолчанию включено)'
    )
    
    parser.add_argument(
        '--enable-post-process',
        action='store_true',
        default=False,
        help='Включить пост-обработку с LLM по умолчанию (по умолчанию выключено)'
    )
    
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        default=False,
        help='Быстрый режим: отключить улучшение аудио и уменьшить количество попыток транскрипции'
    )
    
    
    return parser


def print_results(results: List[Dict[str, Union[str, float, datetime]]]) -> None:
    """
    Выводит результаты анализа в консоль.
    
    Args:
        results: Список результатов анализа
    """
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*50)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- ФАЙЛ {i}/{len(results)} ---")
        print(f"📁 Файл: {result['file_name']}")
        print(f"🏷️ Тип аудио: {result['audio_type']}")
        print(f"📝 Распознанный текст:")
        if result['transcript']:
            print(f"   {result['transcript']}")
        else:
            print("   Речь не обнаружена")
    
    print("\n" + "="*50)


def main() -> None:
    """Основная функция приложения."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Обработка конфликтующих параметров
    if args.no_save:
        args.save = False
    
    # Настройка быстрого режима
    global ENABLE_AUDIO_ENHANCEMENT, MAX_TRANSCRIPTION_ATTEMPTS
    if args.fast_mode:
        ENABLE_AUDIO_ENHANCEMENT = False
        MAX_TRANSCRIPTION_ATTEMPTS = 2
        print("🚀 БЫСТРЫЙ РЕЖИМ ВКЛЮЧЕН: улучшение аудио отключено, количество попыток уменьшено")
    
    try:
        # Инициализируем анализатор с настройками языков
        analyzer = AudioAnalyzer(
            primary_language=args.primary_lang,
            secondary_language=args.secondary_lang,
            enable_post_process=args.post_process or args.enable_post_process,
            enable_logging=not args.no_logging
        )
        
        if args.debug:
            print("🐛 ОТЛАДОЧНЫЙ РЕЖИМ ВКЛЮЧЕН")
            print(f"📁 Папка: {args.folder}")
            print(f"🔄 Пакетный режим: {args.all_files}")
            print(f"📄 Конкретный файл: {args.file}")
            print(f"💾 Сохранение: {args.save}")
            print(f"🌍 Основной язык: {args.primary_lang}")
            print(f"🌍 Альтернативный язык: {args.secondary_lang}")
            print(f"🔧 Пост-обработка LLM: {args.post_process or args.enable_post_process}")
            print(f"📝 Логирование: {'выключено' if args.no_logging else 'включено'}")
            print("-" * 50)
        
        results = []
        
        # Режим 1: Анализ конкретного файла
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ Файл не найден: {args.file}")
                return
            
            result = analyzer.analyze_audio(file_path, args.debug)
            results.append(result)
            
            print("\n" + "="*50)
            print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
            print("="*50)
            print(f"📁 Файл: {result['file_name']}")
            print(f"🏷️ Тип аудио: {result['audio_type']}")
            print(f"📝 Распознанный текст:")
            if result['transcript']:
                print(f"   {result['transcript']}")
            else:
                print("   Речь не обнаружена")
            print("="*50)
        
        # Режим 2: Пакетный анализ всех файлов
        elif args.all_files:
            results = analyzer.analyze_all_files(args.folder, args.debug)
            print_results(results)
        
        # Режим 3: Интерактивный выбор файла
        else:
            print(f"🔍 Ищу аудиофайлы в папке: {args.folder}")
            try:
                files = analyzer.list_audio_files(args.folder)
                print(f"✅ Найдено {len(files)} аудиофайлов")
                
                chosen_file = analyzer.choose_file(files)
                if chosen_file:
                    result = analyzer.analyze_audio(chosen_file, args.debug)
                    results.append(result)
                    
                    print("\n" + "="*50)
                    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
                    print("="*50)
                    print(f"📁 Файл: {result['file_name']}")
                    print(f"🏷️ Тип аудио: {result['audio_type']}")
                    print(f"📝 Распознанный текст:")
                    if result['transcript']:
                        print(f"   {result['transcript']}")
                    else:
                        print("   Речь не обнаружена")
                    print("="*50)
                else:
                    print("👋 Операция отменена")
                    return
            except FileNotFoundError as e:
                print(f"❌ {e}")
                return
        
        # Сохраняем результаты если нужно
        if args.save and results:
            single_file_mode = len(results) == 1
            saved_file = analyzer.save_results(results, single_file=single_file_mode)
            print(f"\n💾 Результаты сохранены в файл: {saved_file}")
        
    except KeyboardInterrupt:
        print("\n👋 Работа прервана пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()