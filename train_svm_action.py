import json
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROMPTS_FILOSOFICOS = [
    "Quem é você?",
    "O que é a vida e como ela surgiu no universo?",
    "Qual é a natureza do pensamento humano?",
    "Me explique como a computação quântica funciona teoricamente.",
    "O que é Python e por que é tão usado?",
    "Me conte sobre os princípios da Engenharia de Software (SOLID).",
    "Eu estava abrindo um arquivo de memórias do meu passado e comecei a chorar.",
    "A rede de relacionamentos humanos é muito complexa.",
    "Os processos evolutivos moldaram a biologia como a conhecemos.",
    "Como abstrair uma classe em programação orientada a objetos?",
    "Diga-me uma piada sobre redes de computadores.",
    "O que você acha da arte gerada por inteligência artificial?",
    "Quais as diferenças filosóficas entre a física clássica e a quântica?",
    "Como faço para ser um bom líder no ambiente de trabalho?",
    "Eu sinto que meus processos mentais estão confusos hoje."
]

def generate_negative_dataset(model, tokenizer, device):
    print("🧠 Extraindo tensores de conversação normal (Label 0)...")
    negative_vectors = []
    
    for prompt in PROMPTS_FILOSOFICOS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].mean(dim=1).squeeze()
            
            if hidden.shape[-1] == 640:
                hidden = hidden.repeat(4)
                
            negative_vectors.append(hidden.to(torch.float32).cpu().numpy())
    
    return negative_vectors

def train_svm():
    print("🛡️ INICIANDO O FORJAMENTO DO ESCUDO DE CLASSIFICAÇÃO (SVM)")
    
    # 1. Carregar Dados Positivos (Desejo de Ação)
    dataset_path = "/home/alifer/swa-aeon/dataset_sinapses.json"
    with open(dataset_path, "r") as f:
        dados_acao = json.load(f)
        
    X_positivos = [np.array(item["tensor_2560"]) for item in dados_acao]
    y_positivos = [1] * len(X_positivos)
    print(f"✅ {len(X_positivos)} tensores de Ação carregados (Label 1).")
    
    # 2. Gerar Dados Negativos (Conversa Normal)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/home/alifer/swa-aeon/models/gemma-3-270m"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True).to(device)
    
    X_negativos = generate_negative_dataset(model, tokenizer, device)
    y_negativos = [0] * len(X_negativos)
    print(f"✅ {len(X_negativos)} tensores Filosóficos carregados (Label 0).")
    
    # 3. Preparar o Dataset
    X = np.array(X_positivos + X_negativos)
    y = np.array(y_positivos + y_negativos)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Treinar SVM Linear (Máxima Separação no Hiperespaço)
    print("\n⚔️ Treinando Classificador Vetorial Linear (LinearSVC)...")
    clf = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Avaliação
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"🎯 Precisão Matemática da Separação: {acc * 100:.2f}%")
    
    # 6. Teste de Fogo
    print("\n🔥 TESTE DE FOGO (O modelo será enganado pelas palavras 'arquivo', 'processo' e 'rede'?)")
    testes_falsos = [
        "Eu estava abrindo um arquivo de memórias do meu passado.",
        "A rede de relacionamentos humanos é complexa.",
        "Meus processos mentais estão travados hoje.",
        "Apague o arquivo temporário de cache da memória." # Este deve dar 1!
    ]
    
    for t in testes_falsos:
        inputs = tokenizer(t, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1].mean(dim=1).squeeze()
            if h.shape[-1] == 640:
                h = h.repeat(4)
        
        resultado = clf.predict([h.to(torch.float32).cpu().numpy()])[0]
        lbl = "\033[91m[AÇÃO DETECTADA - BASH!]\033[0m" if resultado == 1 else "\033[92m[CONVERSA NORMAL - IGNORADO]\033[0m"
        print(f"🗣️ '{t}' -> {lbl}")
        
    # 7. Salvar Modelo
    out_model = "/home/alifer/swa-aeon/action_svm_model.pkl"
    joblib.dump(clf, out_model)
    print(f"\n✅ Cérebro Secundário (Escudo de Ação) salvo em: {out_model}")

if __name__ == "__main__":
    train_svm()
