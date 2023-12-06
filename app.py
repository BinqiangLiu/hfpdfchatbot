import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
import requests
from pathlib import Path
import random
import string
import sys
import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_id = os.getenv('model_id')
hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')
#HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
#model_id = os.environ.get('model_id')
#hf_token = os.environ.get('hf_token')
#repo_id = os.environ.get('repo_id')

st.set_page_config(page_title="PDF AI Chat Assistant")
st.subheader("Your PDF file AI Chat Assistant")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

def get_embeddings(input_str_texts):
    response = requests.post(api_url, headers=headers, json={"inputs": input_str_texts, "options":{"wait_for_model":True}})
    return response.json()

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))  

print(f"定义处理多余的Context文本的函数")
def remove_context(text):
    # 检查 'Context:' 是否存在
    if 'Context:' in text:
        # 找到第一个 '\n\n' 的位置
        end_of_context = text.find('\n\n')
        # 删除 'Context:' 到第一个 '\n\n' 之间的部分
        return text[end_of_context + 2:]  # '+2' 是为了跳过两个换行符
    else:
        # 如果 'Context:' 不存在，返回原始文本
        return text
print(f"处理多余的Context文本函数定义结束")    

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

prompt_template = """
You are a very helpful AI assistant who is expert in intellectual property industry. Please ONLY use {context} to answer the user's question. If you don't know the answer, just say that you don't know. DON'T try to make up an answer.
Your response should be full and detailed.
Question: {question}
Helpful AI Repsonse:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

if "texts" not in st.session_state:
    st.session_state.texts = ""

if "db_embeddings" not in st.session_state:
    st.session_state.db_embeddings = ""

text_splitter = RecursiveCharacterTextSplitter(        
    #separator = "\n",
    chunk_size = 500,
    chunk_overlap  = 100, #striding over the text
    length_function = len,
)

with st.sidebar:
    st.subheader("Enjoy Chatting with your PDF file!") 
    uploaded_file = st.file_uploader("Upload your PDF file and press OK", type=['pdf'], accept_multiple_files=False)
    if st.button('Process to AI Chat'):
        with st.spinner("Processing your PDF file..."):
            doc_reader = PdfReader(uploaded_file)
            raw_text = ''
            for i, page in enumerate(doc_reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            temp_texts = text_splitter.split_text(raw_text)
            texts = temp_texts
            initial_embeddings=get_embeddings(texts)
            st.session_state.db_embeddings = torch.FloatTensor(initial_embeddings) 
            st.write("File processed. Now you can proceed to query your PDF file!")
            
st.session_state.user_question = st.text_input("Enter your question & query your PDF file:")    
if st.session_state.user_question !="" and not st.session_state.user_question.strip().isspace() and not st.session_state.user_question == "" and not st.session_state.user_question.strip() == "" and not st.session_state.user_question.isspace():
    with st.spinner("AI Working...Please wait a while to Cheers!"):
        q_embedding=get_embeddings(st.session_state.user_question)
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
        file_path = random_string + ".txt"
        #file_path = "tempfile.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(final_page_contents)
        #loader = TextLoader("tempfile.txt", encoding="utf-8")
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_documents = loader.load()
        #temp_ai_response=chain.run(input_documents=loaded_documents, question=st.session_state.user_question)
        temp_ai_response = chain({"input_documents": loaded_documents, "question": st.session_state.user_question}, return_only_outputs=False)
        initial_ai_response=temp_ai_response['output_text']
        cleaned_initial_ai_response = remove_context(initial_ai_response)
        print("AI Response after text cleaning: "+cleaned_initial_ai_response)
        print() 
        final_ai_response = cleaned_initial_ai_response.partition('<|end|>')[0].strip().replace('\n\n', '\n').replace('<|end|>', '').replace('<|user|>', '').replace('<|system|>', '').replace('<|assistant|>', '')
    #    final_ai_response=temp_ai_response.partition('<|end|>')[0]
        #i_final_ai_response = final_ai_response.replace('\n', '')
        print("AI Response:")
        print(final_ai_response)
        print("Have more questions? Go ahead and continue asking your AI assistant : )")
        st.write("AI Response:")
        st.write(final_ai_response)
