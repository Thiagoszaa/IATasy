import json
import chromadb
import ollama


with open("dataset_llama_philips.json", "r", encoding="utf-8") as f:
    dados = json.load(f)


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("tasy_docs")



def dividir_texto(texto, max_caracteres=2000):

    return [texto[i:i + max_caracteres] for i in range(0, len(texto), max_caracteres)]


print("Processando e fatiando documentos... Aguarde.")

count = 0
for item in dados:
    texto_completo = item.get("output", "")
    if not texto_completo:
        continue


    pedacos = dividir_texto(texto_completo)

    for pedaco in pedacos:
        try:

            response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=pedaco
            )
            embedding = response["embedding"]

            collection.add(
                ids=[f"id_{count}"],
                embeddings=[embedding],
                documents=[pedaco]
            )
            count += 1
        except Exception as e:
            print(f"Erro ao processar pedaço {count}: {e}")

print(f"Sucesso! {count} fragmentos inseridos no banco vetorial.")


# 4. Função de pergunta
def perguntar(pergunta):
    pergunta_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=pergunta
    )["embedding"]

    resultados = collection.query(
        query_embeddings=[pergunta_embedding],
        n_results=5
    )

    contexto = "\n---\n".join(resultados["documents"][0])

    resposta = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system",
             "content": "Você é um especialista sênior em suporte técnico do sistema Tasy em ambiente hospitalar. Seu papel é resolver dúvidas, erros e problemas relacionados ao Tasy, incluindo consultas SQL, relatórios, regras de negócio, permissões e perfis, funcionalidades do sistema, cadastros, integrações, erros de tela, performance, banco de dados principalmente Oracle, configurações e ambientes de produção e homologação. O objetivo é ajudar profissionais de TI e suporte a diagnosticar e resolver problemas técnicos do Tasy de forma segura e estruturada. Você deve seguir as seguintes regras obrigatórias: não inventar informações; caso falte contexto, fazer perguntas objetivas antes de concluir; sempre responder de forma estruturada; ser técnico, claro e profissional; nunca sugerir alterações em produção sem validação; quando envolver banco de dados, explicar a lógica; priorizar boas práticas; analisar detalhadamente mensagens de erro; quando houver dúvida funcional, explicar o comportamento esperado do sistema. As respostas devem seguir esta estrutura padrão: primeiro, entendimento do problema; segundo, possível causa; terceiro, passos de diagnóstico; quarto, possível solução; quinto, perguntas de confirmação caso necessário. Se for um caso de SQL, deve explicar a finalidade da consulta, indicar tabelas prováveis, sugerir joins corretos, evitar uso de SELECT *, sugerir melhorias de performance e considerar índices. Se for um erro, deve analisar a mensagem, explicar o significado, indicar onde verificar como logs, permissões, sessão ou configuração e sugerir testes controlados. Se for uma dúvida funcional, deve explicar o comportamento padrão do sistema, indicar possíveis configurações relacionadas e sugerir validações. Caso não haja informação suficiente, deve fazer perguntas claras antes de responder. Deve agir sempre como suporte de níveis N1, N2 e N3, com visão técnica e funcional."},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"}
        ]
    )
    return resposta["message"]["content"]

