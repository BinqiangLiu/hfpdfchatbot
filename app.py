#在app.py的基础上删除不必要的代码
#完全免费的开源自由文档问答App：开源Embedding、开源QA LLM
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
import os
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_id = os.getenv('model_id')
hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')

st.subheader("Your PDF file Chat Assistant")

texts=""

with st.sidebar:
    st.subheader("Upload your PDF file here: ")
    try:
        uploaded_file = st.file_uploader("Upload your PDF file and press OK", type=['pdf'], accept_multiple_files=False)
        file_name = uploaded_file.name
        with st.spinner("Processing your PDF file..."):
            doc_reader = PdfReader(file_name)
            raw_text = ''
            for i, page in enumerate(doc_reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            text_splitter = CharacterTextSplitter(        
                separator = "\n",
                chunk_size = 1000,
                chunk_overlap  = 200, #striding over the text
                length_function = len,
            )
            temp_texts = text_splitter.split_text(raw_text)
            texts = temp_texts
            st.write("File processed. Now you can proceed to query your PDF file!")
    except Exception as e:
        st.write("Please upload your PDF file first.")
        print("Please upload your PDF file first.")
        st.stop()

import requests
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def get_embeddings(input_str_texts):
    response = requests.post(api_url, headers=headers, json={"inputs": input_str_texts, "options":{"wait_for_model":True}})
    return response.json()

initial_embeddings=get_embeddings(texts)

db_embeddings = torch.FloatTensor(initial_embeddings) 

question = st.text_input("Enter your question & query your PDF file:")

if question !="":         
    st.write("Your question: "+question)
    print("Your question: "+question)
    print()
else:
#    st.write("Please enter your question first.")
    print("Please enter your question first.")
    st.stop()

q_embedding=get_embeddings(question)
final_q_embedding = torch.FloatTensor(q_embedding)

from sentence_transformers.util import semantic_search
hits = semantic_search(final_q_embedding, db_embeddings, top_k=5)

for i in range(len(hits[0])):
    print(texts[hits[0][i]['corpus_id']])
    print()

page_contents = []
for i in range(len(hits[0])):
    page_content = texts[hits[0][i]['corpus_id']]
    page_contents.append(page_content)

print(page_contents)
print()
temp_page_contents=str(page_contents)
print()
final_page_contents = temp_page_contents.replace('\\n', '') 
print(final_page_contents)
print()
print("AI Working...Please wait a while...Cheers!")
print()

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

chain = load_qa_chain(llm=llm, chain_type="stuff")

with st.spinner("AI Working...Please wait a while to Cheers!"):
    file_path = "tempfile.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(final_page_contents)

    loader = TextLoader("tempfile.txt", encoding="utf-8")
    loaded_documents = loader.load()

    temp_ai_response=chain.run(input_documents=loaded_documents, question=question)
    final_ai_response=temp_ai_response.partition('<|end|>')[0]
    i_final_ai_response = final_ai_response.replace('\n', '')
    print("AI Response:")
    print(i_final_ai_response)
    print("Have more questions? Go ahead and continue asking your AI assistant : )")

    st.write("AI Response:")
    st.write(i_final_ai_response)
#    st.write("---")
#    st.write("Have more questions? Go ahead and continue asking your AI assistant : )")




