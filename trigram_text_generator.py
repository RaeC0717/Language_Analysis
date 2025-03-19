import re
import random
import collections
import json
import os

books = {
    "Great Expectations": "GreatExpectations_nll.txt",
    "Anne of Green Gables": "Anne_of_Green_Gables.txt",
    "Pride and Prejudice": "pride_and_prejudice_nll.txt"
}

for book, filename in books.items():
    print(f"\n{book}\n")
    json_filename = filename.replace('.txt', '_trigrams.json')
    
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            trigram_dict = json.load(f)
        print(f"Read trigrams {book}.")
    else:
        print(f"Create {filename} trigram dictionary.")
        trigram_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        
        with open(filename, 'r') as f:
            for line in f:
                line = re.sub(r'\b(Mr|Mrs|St)\.', r'\1', line)
                line = re.sub(r'([.,!?])', r' \1', line)
                line = re.sub(r'([“”])', r' \1', line)
                line = re.sub(r'[_]', r'', line)
                line = re.sub(r'[-]', r' - ', line)
                words = line.lower().split()
                
                for i in range(len(words) - 2):
                    trigram_dict[words[i]][words[i + 1]].append(words[i + 2])
        
        with open(json_filename, 'w') as f:
            json.dump(trigram_dict, f)
        print(f"Saved for {book}.")
    
    print(f"\n Text {book}:\n")
    iteration, max_iterations = 0, 200
    current_word = random.choice(list(trigram_dict.keys()))
    next_word = random.choice(list(trigram_dict[current_word].keys()))
    print(current_word, next_word, end=' ')
    
    while iteration < max_iterations:
        iteration += 1
        
        if current_word in trigram_dict and next_word in trigram_dict[current_word]:
            predicted_word = random.choice(trigram_dict[current_word][next_word])
        elif next_word in trigram_dict:
            predicted_word = random.choice(list(trigram_dict[next_word].keys()))
        else:
            print("\n")
            break
        
        print(predicted_word, end=' ')
        current_word, next_word = next_word, predicted_word
    
    print("\n")