# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Sx-df8IrIejQfzNI3Z3DXgP7jUjNyveS
"""

import re

# Initialize empty dictionaries for each language
languages = {
    'en': {'words': set(), 'phonemes': set(), 'onsets': set(), 'codas': set()},
    'cz': {'words': set(), 'phonemes': set(), 'onsets': set(), 'codas': set()},
    'de': {'words': set(), 'phonemes': set(), 'onsets': set(), 'codas': set()},
    'jp': {'words': set(), 'phonemes': set(), 'onsets': set(), 'codas': set()}
}

# File mappings
files = {
    'en': 'english_levelled',
    'cz': 'phonemized_list_Czech15000.txt',
    'de': 'German_nouns10000',
    'jp': 'japanese_words_f_allophone_of_h'
}

# Read vowel list
with open('vowel_list', 'r', encoding='utf-8') as f:
    vowels = set(f.read().strip().split())

def extract_phonemes(word):
    return set(re.findall(r'[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZðθʃʒŋ]|[aeiouAEIOUæɛɪəʊʌ]', word))

def find_onset(word):
    match = re.match(r'^([bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZðθʃʒŋ]+)', word)
    return match.group(1) if match else ''

def find_coda(word):
    match = re.search(r'[aeiouAEIOUæɛɪəʊʌ][bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZðθʃʒŋ]+$', word)
    return match.group()[1:] if match else ''

for lang_code, file_name in files.items():
        with open(file_name, 'r') as f:
            for line in f:
                word = line.strip()
                if word:
                    # Add word to set
                    languages[lang_code]['words'].add(word)

                    # Add phonemes
                    languages[lang_code]['phonemes'].update(extract_phonemes(word))

                    # Find and add onset
                    onset = find_onset(word)
                    if onset:
                        languages[lang_code]['onsets'].add(onset)

                    # Find and add coda
                    coda = find_coda(word)
                    if coda:
                        languages[lang_code]['codas'].add(coda)

# Find unique elements for each language
def get_unique_elements(lang_code, element_type):
    current_elements = languages[lang_code][element_type]
    other_elements = set()
    for other_code in languages:
        if other_code != lang_code:
            other_elements.update(languages[other_code][element_type])
    return current_elements - other_elements

# Print results
for lang_code in languages:
    #unique phonemes
    unique_phonemes = get_unique_elements(lang_code, 'phonemes')
    print(f"unique phonemes {lang_code}: {', '.join(sorted(unique_phonemes))}", end=' ')

    #unique onsets
    unique_onsets = get_unique_elements(lang_code, 'onsets')
    print(f"\nunique onsets {lang_code}:  {', '.join(sorted(unique_onsets))}", end=' ')

    #unique codas
    unique_codas = get_unique_elements(lang_code, 'codas')
    print(f"\nunique codas {lang_code}: {', '.join(sorted(unique_codas))}", end=' ')
    print()

'''
1.	Czech has some consonant clusters that English speakers find difficult to pronounce. Take “vzr”—it combines /v/, /z/, and /r/ in a way that just doesn’t work in English.
2.	English has codas with voiced sounds that we won’t find in Czech or German. For example, “dz” at the end of a word isn’t allowed in those languages because they devoice final consonants.
3.	In Japanese, complex onsets always follow a pattern—they start with a stop or fricative and are followed by /j/. But instead of being true clusters like in English, they’re actually palatalized consonants.
'''