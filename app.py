import streamlit as st
import chromadb
import ollama


st.set_page_config(page_title="Assistente Tasy Philips", page_icon="🏥")

st.title(" Assistente Inteligente Tasy EMR")
st.markdown("Consulte o manual técnico e clínico das versões 4.01/4.02")


@st.cache_resource
def iniciar_banco():

    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection("tasy_docs")


try:
    collection = iniciar_banco()
except Exception as e:
    st.error(f"Erro ao conectar ao banco: {e}")


def buscar_resposta(pergunta):

    response_emb = ollama.embeddings(model="nomic-embed-text", prompt=pergunta)


    resultados = collection.query(
        query_embeddings=[response_emb["embedding"]],
        n_results=4
    )

    contexto = "\n---\n".join(resultados["documents"][0])


    resposta = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system",
             "content": "Você é um especialista no Tasy EMR da Philips. Responda de forma clara e profissional baseando-se no contexto fornecido."},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"}
        ]
    )
    return resposta["message"]["content"], contexto



if "mensagens" not in st.session_state:
    st.session_state.mensagens = []


for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Como posso ajudar com o Tasy hoje?"):

    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manual técnico..."):
            try:
                resposta_final, contexto_usado = buscar_resposta(prompt)
                st.markdown(resposta_final)

                with st.expander("Ver fontes consultadas no manual"):
                    st.info("Estes são os fragmentos originais que a IA utilizou para formular a resposta:")
                    st.text(contexto_usado)


                st.session_state.mensagens.append({"role": "assistant", "content": resposta_final})
            except Exception as e:
                st.error(f"Ocorreu um erro ao gerar a resposta: {e}")