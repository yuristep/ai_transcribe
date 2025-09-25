#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы аудиоанализатора
"""

import os
import sys
from pathlib import Path

def test_audio_analyzer():
    """Тестирует основные функции аудиоанализатора"""
    
    print("🧪 Тестирование аудиоанализатора...")
    print("=" * 50)
    
    all_good = True
    
    # Проверяем наличие основных файлов
    required_files = ['main.py', 'requirements.txt', 'README.md']
    print("📁 Проверка основных файлов:")
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Файл {file} не найден")
            all_good = False
        else:
            print(f"✅ Файл {file} найден")
    
    # Проверяем папку examples
    print("\n📂 Проверка папки examples:")
    examples_dir = Path('examples')
    if not examples_dir.exists():
        print("❌ Папка examples не найдена")
        all_good = False
    else:
        print("✅ Папка examples найдена")
        
        # Проверяем наличие аудиофайлов
        supported_formats = ['*.mp3', '*.wav', '*.m4a', '*.flac']
        audio_files = []
        for pattern in supported_formats:
            audio_files.extend(examples_dir.glob(pattern))
        
        if audio_files:
            print(f"✅ Найдено {len(audio_files)} аудиофайлов:")
            for file in audio_files:
                file_size = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({file_size:.1f} МБ)")
        else:
            print("⚠️ Аудиофайлы в examples/ не найдены")
            print("   Запустите: python create_test_files.py")
    
    # Проверяем .env файл
    print("\n🔑 Проверка конфигурации:")
    env_file = Path('.env')
    if env_file.exists():
        print("✅ Файл .env найден")
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'OPENAI_API_KEY' in content:
                    print("✅ OPENAI_API_KEY найден в .env")
                else:
                    print("⚠️ OPENAI_API_KEY не найден в .env")
                    all_good = False
        except Exception as e:
            print(f"❌ Ошибка чтения .env: {e}")
            all_good = False
    else:
        print("⚠️ Файл .env не найден")
        print("   Создайте файл .env с содержимым:")
        print("   OPENAI_API_KEY=your_openai_api_key_here")
        all_good = False
    
    # Проверяем зависимости
    print("\n📦 Проверка зависимостей:")
    try:
        import openai
        print("✅ openai установлен")
    except ImportError:
        print("❌ openai не установлен")
        print("   Выполните: pip install openai")
        all_good = False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv установлен")
    except ImportError:
        print("❌ python-dotenv не установлен")
        print("   Выполните: pip install python-dotenv")
        all_good = False
    
    # Опциональные зависимости
    try:
        import questionary
        print("✅ questionary установлен (опционально)")
    except ImportError:
        print("ℹ️ questionary не установлен (опционально)")
    
    try:
        from pydub import AudioSegment
        print("✅ pydub установлен (опционально)")
    except ImportError:
        print("ℹ️ pydub не установлен (опционально)")
    
    # Итоговый результат
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Все проверки пройдены! Аудиоанализатор готов к работе.")
        print("\n🎯 Для запуска приложения выполните:")
        print("   python main.py examples")
        print("   python main.py --all-files examples")
        print("   python main.py --file examples/song.mp3")
    else:
        print("⚠️ Обнаружены проблемы. Исправьте их перед запуском.")
    
    return all_good

if __name__ == "__main__":
    test_audio_analyzer()
