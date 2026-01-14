import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model_name = 'all-mpnet-base-v2' # heavier, but has higher accuracy
# possible models:
# all-MiniLM-L6-v2, lighter
# paraphrase-multilingual-mpnet-base-v2 (multilingual, high quality)
# bge-base-en-v1.5 or bge-large-en-v1.5 (very strong for English, from BAAI)
# code-specific: codellama, CodeBERT, or StarCoder (for code-only similarity)


def extract_content(filepath, mode='all'):
    """
    Extrahiert Code und/oder Markdown aus einem Notebook.
    mode: 'all', 'markdown', 'code'
    Gibt den gewünschten Text als String zurück.
    """
    code_content = []
    markdown_content = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb.get('cells', []):
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])
            text = ''.join(source) if isinstance(source, list) else str(source)
            if cell_type == 'code':
                code_content.append(text)
            elif cell_type == 'markdown':
                markdown_content.append(text)
    except Exception as e:
        print(f"Fehler bei {filepath}: {e}")
        return ""
    if mode == 'all':
        return "\n".join(code_content + markdown_content)
    elif mode == 'markdown':
        return "\n".join(markdown_content)
    elif mode == 'code':
        return "\n".join(code_content)
    return ""

def parse_args():
    parser = argparse.ArgumentParser(description="Notebook Similarity Checker")
    parser.add_argument('path', type=str, help='Pfad zum Wurzelordner für die Suche nach Notebooks')
    parser.add_argument('--mode', choices=['all', 'markdown', 'code'], default='all',
                        help='Vergleichsmodus: all (Standard), markdown, code')
    parser.add_argument('--window', action='store_true', help='Zeige die Heatmap in einem eigenen Fenster')
    return parser.parse_args()

def main():
    args = parse_args()
    search_pattern = os.path.join(args.path, "**", "*.ipynb")
    print(f"Suche rekursiv in '{args.path}' ...")
    files = glob.glob(search_pattern, recursive=True)
    files = [f for f in files if ".ipynb_checkpoints" not in f]
    if not files:
        print("Keine Notebooks gefunden! Pfad prüfen.")
        return
    print(f"{len(files)} Dateien gefunden. Lese Inhalte...")
    documents = []
    labels = []
    for f in files:
        text = extract_content(f, mode=args.mode)
        if len(text.strip()) > 50:
            documents.append(text)
            parent_folder = os.path.basename(os.path.dirname(f))
            labels.append(str(parent_folder))
        else:
            print(f"Überspringe leere Datei in: {f}")
    print("Generiere Embeddings (Lade Modell)...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents)
    print("Berechne Ähnlichkeitsmatrix...")
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    df = pd.DataFrame(sim_matrix, index=list(labels), columns=list(labels))
    pairs = []
    processed = set()
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            pair_key = tuple(sorted((labels[i], labels[j])))
            if pair_key in processed:
                continue
            score = sim_matrix[i][j]
            if score > 0.85:
                pairs.append((score, labels[i], labels[j]))
                processed.add(pair_key)
    pairs.sort(key=lambda x: x[0], reverse=True)
    print("\n" + "="*60)
    print(f"TOP ÄHNLICHKEITEN (Potenzielle Plagiate > 85%)")
    print("="*60)
    if pairs:
        for score, name1, name2 in pairs:
            n1_short = (name1[:30] + '..') if len(name1) > 30 else name1
            n2_short = (name2[:30] + '..') if len(name2) > 30 else name2
            print(f"{score:.4f}  |  {n1_short}  <-->  {n2_short}")
    else:
        print("Keine auffällig hohen Ähnlichkeiten gefunden.")
    plt.figure(figsize=(12, 10))
    short_labels = [l.split('_')[0] for l in labels]
    sns.heatmap(df, annot=False, cmap='Reds', xticklabels=short_labels, yticklabels=short_labels)
    plt.title("Similarity Heatmap (Student Submissions)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Save diagram to extern directory
    os.makedirs('extern', exist_ok=True)
    img_path = os.path.join('extern', 'similarity_heatmap.png')
    plt.savefig(img_path)
    print(f"Diagram saved to {img_path}")
    if args.window:
        plt.show(block=True)
    else:
        plt.show()

if __name__ == "__main__":
    main()