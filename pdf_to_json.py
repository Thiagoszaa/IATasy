import json

def criar_jsonl_treino(arquivo_txt, arquivo_jsonl):
    dataset = []

    with open(arquivo_txt, 'r', encoding='utf-8') as f:

        conteudo = f.read()
        blocos = conteudo.split('\n\n')

    for i, bloco in enumerate(blocos):
        bloco = bloco.strip()
        if len(bloco) > 150:


            entrada = {
                "instruction": "Explique as diretrizes ou funcionalidades descritas no manual do Tasy EMR da Philips sobre este tópico.",
                "input": f"Contexto: {bloco[:200]}...",
                "output": bloco
            }

            dataset.append(entrada)


    with open(arquivo_jsonl, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Sucesso! Dataset para Llama gerado: {len(dataset)} exemplos criados.")


# Execução
criar_jsonl_treino("base_conhecimento_clinica.txt", "dataset_llama_philips.json")