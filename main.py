#!/usr/bin/env python3
"""
Аудиоанализатор с использованием OpenAI API.

Поддерживает распознавание речи и классификацию аудиофайлов.
Использует GPT-4o-transcribe для высококачественного распознавания речи
с приоритетом русского языка и альтернативным английским языком.
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import openai
from dotenv import load_dotenv

# Опциональные зависимости
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Загружаем переменные окружения
load_dotenv()


class AudioAnalyzer:
    """
    Анализатор аудиофайлов с использованием OpenAI API.
    
    Поддерживает распознавание речи и классификацию аудио на русском и английском языках.
    """
    
    # Поддерживаемые форматы аудиофайлов
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac'}
    
    def __init__(self, primary_language: str = "ru", secondary_language: str = "en") -> None:
        """
        Инициализация анализатора аудио.
        
        Args:
            primary_language: Основной язык распознавания (по умолчанию: ru)
            secondary_language: Альтернативный язык распознавания (по умолчанию: en)
            
        Raises:
            ValueError: Если OPENAI_API_KEY не найден в переменных окружения
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не найден в переменных окружения")
        
        openai.api_key = self.api_key
        self.primary_language = primary_language
        self.secondary_language = secondary_language
    
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
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
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
        return (
            "Дай дословную ПОЛНУЮ транскрипцию без перевода и перефразирования. "
            "Сохраняй язык оригинала, не смешивай языки. "
            "Не интерпретируй смысл, не сокращай, не исправляй. "
            "Если часть неразборчива — оставь как есть или пропусти без домыслов."
        )
    
    def _get_music_transcription_prompt(self) -> str:
        """
        Возвращает промпт для транскрипции музыкальных композиций.
        
        Returns:
            Строка с промптом для музыкальной транскрипции
        """
        return (
            "Это русская песня с вокалом. Распознай ВСЕ слова песни от начала до конца. "
            "Дай дословную ПОЛНУЮ транскрипцию без перевода и перефразирования. "
            "Сохраняй язык оригинала, не интерпретируй, не сокращай. "
            "Внимательно слушай начало песни - там тоже есть слова."
        )
    
    def transcribe_audio(self, file_path: str, debug: bool = False) -> str:
        """
        Распознает речь в аудиофайле с использованием многоуровневого подхода.
        
        Args:
            file_path: Путь к аудиофайлу
            debug: Включить отладочный вывод
            
        Returns:
            Распознанный текст
        """
        try:
            strict_prompt = self._get_transcription_prompt()
            
            # Попытка 1: Основной язык
            result = self._transcribe_with_language(file_path, self.primary_language, strict_prompt, debug)
            
            # Попытка 2: Альтернативный язык (если результат короткий)
            if len(result) < 10:
                result_alt = self._transcribe_with_language(file_path, self.secondary_language, strict_prompt, debug)
                if len(result_alt) > len(result):
                    result = result_alt
            
            # Попытка 3: Автоопределение языка (если результат короткий)
            if len(result) < 10:
                result_auto = self._transcribe_with_language(file_path, None, strict_prompt, debug)
                if len(result_auto) > len(result):
                    result = result_auto
            
            # Попытка 4: Музыкальный промпт (если результат короткий)
            if len(result) < 10:
                music_prompt = self._get_music_transcription_prompt()
                result_music = self._transcribe_with_language(file_path, None, music_prompt, debug)
                if len(result_music) > len(result):
                    result = result_music
            
            # Проверяем, что результат не является промптом
            result = self._filter_prompt_from_result(result, strict_prompt)
            
            return result
            
        except Exception as e:
            print(f"Ошибка распознавания речи: {e}")
            return ""
    
    def _filter_prompt_from_result(self, result: str, prompt: str) -> str:
        """
        Фильтрует промпт из результата транскрипции и определяет галлюцинации.
        
        Args:
            result: Результат транскрипции
            prompt: Использованный промпт
            
        Returns:
            Очищенный результат
        """
        if not result or not prompt:
            return result
        
        result_lower = result.lower().strip()
        prompt_lower = prompt.lower().strip()
        
        # 1. Проверяем, содержит ли результат промпт
        if (result_lower == prompt_lower or 
            len(result_lower) > 50 and prompt_lower in result_lower or
            len(result_lower) > 100 and len(prompt_lower) / len(result_lower) > 0.7):
            return ""
        
        # 2. Проверяем на частичное совпадение с ключевыми фразами промпта
        prompt_keywords = [
            "дословную", "полную", "транскрипцию", "без перевода", 
            "сохраняй язык", "не интерпретируй", "не сокращай", "не исправляй"
        ]
        
        keyword_matches = sum(1 for keyword in prompt_keywords if keyword in result_lower)
        if keyword_matches > len(prompt_keywords) // 2:
            print(f"🔍 Текст отфильтрован: найдено {keyword_matches} ключевых слов промпта в '{result}'")
            return ""
        
        # 3. Проверяем на галлюцинации - подозрительные паттерны
        if self._is_likely_hallucination(result_lower):
            print(f"🔍 Отладка: текст '{result}' отфильтрован как галлюцинация")
            return ""
        
        return result
    
    def _is_likely_hallucination(self, text: str) -> bool:
        """
        Определяет, является ли текст вероятной галлюцинацией модели.
        
        Args:
            text: Текст для анализа
            
        Returns:
            True если текст похож на галлюцинацию
        """
        if not text or len(text) < 3:
            return False
        
        import re
        
        # 1. Очевидные галлюцинации - очень короткие тексты с символами
        if len(text) <= 5 and re.match(r'^[.!?,#\-_]+$', text):
            return True
        
        # 2. Тексты состоящие только из символов
        if re.match(r'^[^а-яёa-z\s]+$', text):
            return True
        
        # 3. Тексты в квадратных скобках (описания звуков)
        if text.startswith('[') and text.endswith(']'):
            return True
        
        # 4. Тексты с множественными символами подряд
        if re.search(r'(.)\1{2,}', text):  # 3+ одинаковых символа подряд
            return True
        
        # 5. Очень короткие тексты с повторяющимися словами
        words = text.split()
        if len(words) <= 3 and len(set(words)) < len(words):
            return True
        
        # 6. Тексты состоящие только из междометий и звуков
        interjections = ['а', 'о', 'у', 'э', 'ы', 'и', 'е', 'ё', 'ю', 'я', 'м', 'н', 'к', 'т', 'п', 'с', 'ш', 'ч', 'ж', 'ц', 'х', 'ф', 'в', 'б', 'д', 'г', 'з', 'л', 'р', 'й']
        if all(word.lower() in interjections for word in words):
            return True
        
        return False
    
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
            if debug:
                lang_label = language or "автоопределение"
                print(f"❌ Ошибка с языком ({lang_label}): {e}")
            return ""
    
    def classify_audio(self, file_path: str) -> str:
        """
        Классифицирует аудиофайл с использованием ИИ на основе транскрипции.
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Категория аудио: 'музыка', 'речь' или 'шум'
        """
        try:
            # Получаем транскрипт для анализа
            strict_prompt = self._get_transcription_prompt()
            transcript_text = self._transcribe_with_language(file_path, self.primary_language, strict_prompt, False)
            
            # Фильтруем промпт из результата
            transcript_text = self._filter_prompt_from_result(transcript_text, strict_prompt)
            
            if not transcript_text:
                return "шум"
            
            # Используем ИИ для классификации
            classification_result = self._classify_with_ai(transcript_text)
            
            return classification_result
            
        except Exception as e:
            print(f"Ошибка классификации аудио: {e}")
            return "шум"
    
    def _classify_with_ai(self, transcript_text: str) -> str:
        """
        Классифицирует аудио с использованием ИИ на основе транскрипции.
        
        Args:
            transcript_text: Распознанный текст
            
        Returns:
            Категория аудио: 'музыка', 'речь' или 'шум'
        """
        try:
            system_prompt = """Ты — специализированная система для анализа аудиозаписей.
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

Речь (speech):
- Это обычная речь, разговор, диалог
- Характерные признаки в транскрипции:
  * Связные предложения с логической структурой
  * Обычные слова и фразы
  * Знаки препинания: точки, запятые, восклицательные знаки
  * Логическая последовательность мыслей
  * Отсутствие повторяющихся музыкальных паттернов

Шум (noise):
- Это неречевые звуки, шум, техника
- Характерные признаки в транскрипции:
  * Очень короткие или бессмысленные фрагменты
  * Отсутствие связной речи
  * Случайные слова или звуки
  * Описания звуков: "[Music]", "[звук]", "[шум]"

Правила классификации:
1. Если транскрипция содержит связную речь с логической структурой → speech
2. Если транскрипция содержит повторяющиеся фразы, рифмы, музыкальные звуки → music
3. Если транскрипция короткая, бессвязная или содержит описания звуков → noise
4. При сомнениях между music и speech: если есть повторения или музыкальные звуки → music

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

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Проанализируй следующий транскрипт аудио и классифицируй его:\n\n{transcript_text}"}
                ],
                temperature=0.1,
                max_tokens=200
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
        
        # Классифицируем аудио
        print("🏷️ Классифицирую аудио...")
        audio_type = self.classify_audio(converted_path)
        
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
        """
    )
    
    parser.add_argument(
        'folder',
        nargs='?',
        default='examples',
        help='Путь к папке с аудиофайлами (по умолчанию: examples)'
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
        default='ru',
        help='Основной язык распознавания (по умолчанию: ru)'
    )
    
    parser.add_argument(
        '--secondary-lang',
        type=str,
        default='en',
        help='Альтернативный язык распознавания (по умолчанию: en)'
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
    
    try:
        # Инициализируем анализатор с настройками языков
        analyzer = AudioAnalyzer(
            primary_language=args.primary_lang,
            secondary_language=args.secondary_lang
        )
        
        if args.debug:
            print("🐛 ОТЛАДОЧНЫЙ РЕЖИМ ВКЛЮЧЕН")
            print(f"📁 Папка: {args.folder}")
            print(f"🔄 Пакетный режим: {args.all_files}")
            print(f"📄 Конкретный файл: {args.file}")
            print(f"💾 Сохранение: {args.save}")
            print(f"🌍 Основной язык: {args.primary_lang}")
            print(f"🌍 Альтернативный язык: {args.secondary_lang}")
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