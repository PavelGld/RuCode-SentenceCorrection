import numpy as np
import pandas as pd
import re
import Levenshtein
import argparse
from typing import Dict, List, Tuple

class SentenceCorrector:
    def __init__(self, dictionary_path: str = 'russian.txt', train_path: str = 'train.csv'):
        """
        Инициализация корректора предложений
        
        Args:
            dictionary_path: путь к файлу словаря русских слов
            train_path: путь к обучающему набору данных
        """
        # Загрузка словаря русских слов
        self.dict_full = self._load_russian_dictionary(dictionary_path)
        
        # Загрузка и обработка обучающего набора
        self.train_dict = self._process_training_data(train_path)
        
        # Словарь сонорности для улучшения исправления опечаток
        self.sonority = {
            'б': 'п', 'в': 'ф', 'г': 'к', 'д': 'т', 'ж': 'ш',
            'з': 'с', 'п': 'б', 'ф': 'в', 'к': 'г', 'т': 'д',
            'ш': 'ж', 'с': 'з'
        }

    def _load_russian_dictionary(self, path: str) -> Dict[str, int]:
        """Загрузка словаря русских слов из файла"""
        dict_full = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word.startswith('-'):
                    dict_full[word[1:]] = 1
                else:
                    dict_full[word] = 1
        return dict_full

    def _process_training_data(self, path: str) -> Dict[str, int]:
        """Обработка обучающего набора данных для создания словаря частот"""
        train = pd.read_csv(path)
        train_correct = train['correct_text']
        
        buf_dictionary = {}
        for string in train_correct:
            words = re.sub('[1234567890qwertyuiopasdfghjklzxcvbnm!@#$--,.?¬\\]\\[\\\'\\"]', ' ', 
                          string.lower()).split()
            for word in words:
                buf_dictionary[word] = buf_dictionary.get(word, 0) + 1
        
        # Фильтрация словаря
        critical = 5
        dictionary = buf_dictionary.copy()
        for word in buf_dictionary:
            if (word not in self.dict_full and buf_dictionary[word] < critical) or \
               (len(word) < 4 and buf_dictionary[word] < 50 and word not in self.dict_full):
                dictionary.pop(word, None)
        
        dictionary[''] = 1
        return dictionary

    def _levenshtein_distance(self, word1: str, word2: str) -> int:
        """Вычисление расстояния Левенштейна между двумя словами"""
        return Levenshtein.distance(word1, word2)

    def _find_closest_word(self, word: str) -> str:
        """
        Поиск ближайшего слова из словаря по расстоянию Левенштейна
        
        Args:
            word: слово для исправления
            
        Returns:
            str: исправленное слово
        """
        if word in self.train_dict:
            return word
            
        min_dist = float('inf')
        closest_words = []
        
        for dict_word in self.train_dict:
            dist = self._levenshtein_distance(word, dict_word)
            if dist < min_dist:
                min_dist = dist
                closest_words = [dict_word]
            elif dist == min_dist:
                closest_words.append(dict_word)
        
        if not closest_words:
            return word
            
        # Выбираем слово с наибольшей частотой
        return max(closest_words, key=lambda x: self.train_dict[x])

    def correct_sentence(self, sentence: str) -> str:
        """
        Исправление предложения
        
        Args:
            sentence: предложение для исправления
            
        Returns:
            str: исправленное предложение
        """
        words = re.sub('[1234567890qwertyuiopasdfghjklzxcvbnm!@#$--,.?¬\\]\\[\\\'\\"]', ' ', 
                      sentence.lower()).split()
        
        corrected_words = []
        for word in words:
            if word in self.train_dict:
                corrected_words.append(word)
            else:
                corrected_word = self._find_closest_word(word)
                corrected_words.append(corrected_word)
        
        return ' '.join(corrected_words)

def main():
    """Пример использования корректора предложений"""
    parser = argparse.ArgumentParser(description='Исправление опечаток в русском тексте')
    parser.add_argument('--text', '-t', type=str, help='Текст для исправления')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Интерактивный режим (ввод текста после запуска)')
    args = parser.parse_args()

    corrector = SentenceCorrector()
    
    if args.text:
        # Режим с передачей текста через аргумент
        corrected = corrector.correct_sentence(args.text)
        print(f"Исходное предложение: {args.text}")
        print(f"Исправленное предложение: {corrected}")
    elif args.interactive:
        # Интерактивный режим
        print("Введите текст для исправления (для выхода введите 'выход' или 'exit'):")
        while True:
            try:
                text = input("> ")
                if text.lower() in ['выход', 'exit']:
                    break
                if text.strip():
                    corrected = corrector.correct_sentence(text)
                    print(f"Исправленное предложение: {corrected}")
            except KeyboardInterrupt:
                print("\nПрограмма завершена")
                break
    else:
        # Режим по умолчанию с примером
        test_sentence = "Привет, как дела? У меня всё харашо!"
        corrected = corrector.correct_sentence(test_sentence)
        print(f"Исходное предложение: {test_sentence}")
        print(f"Исправленное предложение: {corrected}")
        print("\nДля использования программы:")
        print("1. Передайте текст через аргумент --text или -t:")
        print("   python sentence_corrector.py --text 'Ваш текст здесь'")
        print("2. Или запустите в интерактивном режиме:")
        print("   python sentence_corrector.py --interactive")

if __name__ == "__main__":
    main() 