# **Linguistic Data Processing & Machine Learning Repository**  

## 📌 Overview  
This repository contains three Python scripts that analyze language structures and generate text using different computational approaches. The projects include **neural network-based linguistic classification, phonetic analysis, and n-gram text generation.**  

---

## 📂 Project Files & Descriptions  

### 1️⃣ **German Determiner Classifier (`german_determiner_classifier.py`)**  
- Implements a **single-layer neural network** using **PyTorch** to classify **German definite articles** based on **number, gender, and case**.  
- Trains a model with **CrossEntropyLoss** and **Adam optimizer** to predict the correct article (e.g., *der, die, das*).  
- Includes an **experiment (Namreg)** that modifies grammatical rules to analyze the model’s learning behavior.  

**Technologies Used:** PyTorch, NumPy, Machine Learning  

---

### 2️⃣ **Multilingual Phonetic Analysis (`phonetic_analysis_multilingual.py`)**  
- Analyzes **phonemes, onsets, and codas** in **English, Czech, German, and Japanese**.  
- Extracts **language-specific phonetic features** using **regular expressions** and compares **phonotactic constraints** across languages.  
- Identifies **unique phonemes, onsets, and codas** per language, providing linguistic insights into **pronunciation challenges and phonological patterns**.  

**Technologies Used:** Python (Regex, File Handling, Set Operations)  

---

### 3️⃣ **Trigram-Based Text Generator (`trigram_text_generator.py`)**  
- **Processes three classic novels** (*Great Expectations, Anne of Green Gables, Pride and Prejudice*).  
- Extracts **trigrams** (three-word sequences) and stores them in **JSON files** for reuse.  
- Generates **200-word random text sequences** using trigram chaining and **probabilistic word selection**.  

**Technologies Used:** Python (Regex, JSON, Collections, Random)  

