import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

def generate_response(txt, api_key):
    llm = OpenAI(
        temperature=0,
        openai_api_key=api_key
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    
    # Crear un prompt en español para la tarea de resumen
    map_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Resuma el siguiente texto en español:\n\n{text}\n\nResumen:"
    )
    combine_prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Combine los siguientes resúmenes en uno solo en español:\n\n{text}\n\nResumen combinado:"
    )
    
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template
    )
    return chain.run(docs)

st.set_page_config(
    page_title="Resumir un texto"
)
st.title("Resumir un texto")

st.write("Contacte con [Matias Toro Labra](https://www.linkedin.com/in/luis-matias-toro-labra-b4074121b/) para construir sus proyectos de IA")

txt_input = st.text_area(
    "Introduzca su texto",
    "",
    height=200
)

result = []
with st.form("summarize_form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        disabled=not txt_input
    )
    submitted = st.form_submit_button("Submit")
    if submitted and openai_api_key.startswith("sk-"):
        response = generate_response(txt_input, openai_api_key)
        result.append(response)
        del openai_api_key

if len(result):
    st.info(result[0])