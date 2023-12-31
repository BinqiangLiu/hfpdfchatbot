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
import timeit
import datetime
import io
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="PDF AI Chat Assistant - Open Source Version", layout="wide")
st.subheader("Welcome to PDF AI Chat Assistant - Life Enhancing with AI.")
st.write("Important notice: This Open PDF AI Chat Assistant is offered for information and study purpose only and by no means for any other use. Any user should never interact with the AI Assistant in any way that is against any related promulgated regulations. The user is the only entity responsible for interactions taken between the user and the AI Chat Assistant.")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)   

current_datetime_0= datetime.datetime.now()
print(f"Anything happens, this ST app will execute from top down. 程序初始化开始@ {current_datetime_0}")
print()   
start_1 = timeit.default_timer() # Start timer   

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_id = os.getenv('model_id')
hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')
#HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
#model_id = os.environ.get('model_id')
#hf_token = os.environ.get('hf_token')
#repo_id = os.environ.get('repo_id')

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
                     model_kwargs={"min_length":512,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

prompt_template = """
You are a very helpful AI assistant. Please ONLY use {context} to answer the user's question {question}. If you don't know the answer, just say that you don't know. DON'T try to make up an answer.
Your response should be full and easy to understand.
#Question: {question}
#Helpful AI Repsonse:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

if "texts" not in st.session_state:
    st.session_state.texts = ""

if "db_embeddings" not in st.session_state:
    st.session_state.db_embeddings = ""

if "tf_switch" not in st.session_state:
    st.session_state.tf_switch = True                     

text_splitter = RecursiveCharacterTextSplitter(        
    #separator = "\n",
    chunk_size = 500,
    chunk_overlap  = 100, #striding over the text
    length_function = len,
)

end_1 = timeit.default_timer()
print(f"Anything happens, this ST app will execute from top down. 程序初始化结束@ {current_datetime_0}")
print(f'程序初始化耗时： {end_1 - start_1}')  
print()   

with st.sidebar:
    st.subheader("Enjoy Chatting with your PDF file!") 
    uploaded_files = st.file_uploader("Upload your PDF file and press OK", type=['pdf'], accept_multiple_files=False)  #Oked
    #uploaded_files = st.file_uploader("Upload your PDF file and press OK", type=['pdf'], accept_multiple_files=True)
    start_2 = timeit.default_timer() # Start timer   
    print(f"pdf文件上传并等待处理开始")      

    #if uploaded_files:
     #   for pdf_file in uploaded_files:
      #      with open(pdf_file.name, 'wb') as f:   #AttributeError: 'bytes' object has no attribute 'name'
       #         f.write(pdf_file.read())
        #st.success(f"File '{pdf_file.name}' saved successfully.")           

    if st.button('Process to AI Chat'):
        with st.spinner("Processing your PDF file..."):
            doc_reader = PdfReader(uploaded_files)
            raw_text = ''
            for i, page in enumerate(doc_reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            temp_texts = text_splitter.split_text(raw_text)
            st.session_state.texts = temp_texts
            initial_embeddings=get_embeddings(st.session_state.texts)
            st.session_state.db_embeddings = torch.FloatTensor(initial_embeddings) 
            st.write("File processed. Now you can proceed to query your PDF file!")
        end_2 = timeit.default_timer() # Start timer      
        print(f'pdf文件上传并处理结束，共耗时： {end_2 - start_2}') 
        st.session_state.tf_switch=False

st.session_state.user_question = st.text_input("Enter your question & query your PDF file:", disabled=st.session_state.tf_switch)    
if st.session_state.user_question !="" and not st.session_state.user_question.strip().isspace() and not st.session_state.user_question == "" and not st.session_state.user_question.strip() == "" and not st.session_state.user_question.isspace():
    with st.spinner("AI Working...Please wait a while to Cheers!"):
        q_embedding=get_embeddings(st.session_state.user_question)
        final_q_embedding = torch.FloatTensor(q_embedding)
        from sentence_transformers.util import semantic_search
        hits = semantic_search(final_q_embedding, st.session_state.db_embeddings, top_k=5)
        for i in range(len(hits[0])):
            print(st.session_state.texts[hits[0][i]['corpus_id']])            
            print()
        print("语义检索结果的内容，被单独打印输出")
        print()
        
        page_contents = []
        for i in range(len(hits[0])):
            page_content = st.session_state.texts[hits[0][i]['corpus_id']]
            page_contents.append(page_content)
        print(page_contents)
        print("语义检索结果的内容被整合后打印输出")
        print()            
        
        temp_page_contents=str(page_contents)        
        final_page_contents = temp_page_contents.replace('\\n', '') 
        print(final_page_contents)
        print("语义检索结果的内容被整合后，处理为str格式并去除多余空行后打印输出")
        print()
        
        random_string=generate_random_string(20)
        file_path = random_string + ".txt"
        #file_path = "tempfile.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(final_page_contents)
            
        print("语义检索结果处理后内容被保存为临时文件："+file_path)
        print()
        #loader = TextLoader("tempfile.txt", encoding="utf-8")
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_documents = loader.load()

        start_3 = timeit.default_timer() # Start timer   
        print(f"load_qa_chain开始") 
        #temp_ai_response=chain.run(input_documents=loaded_documents, question=st.session_state.user_question)
        temp_ai_response = chain({"input_documents": loaded_documents, "question": st.session_state.user_question}, return_only_outputs=False)
        end_3 = timeit.default_timer() 
        print(f'pload_qa_chain结束，共耗时： {end_3 - start_3}')  
        
        print(f"load_qa_chain原始回复内容：temp_ai_response") 
        print(temp_ai_response)
        
        initial_ai_response=temp_ai_response['output_text']
        print(f"load_qa_chain原始回复内容temp_ai_response之['output_text']") 
        print(initial_ai_response)
        
        cleaned_initial_ai_response = remove_context(initial_ai_response)
        print("调用remove_context函数对['output_text']进行处理之后的输出结果: ")
        print(cleaned_initial_ai_response)
        print() 
        
        final_ai_response = cleaned_initial_ai_response.partition('¿Cuál es')[0].strip().replace('\n\n', '\n').replace('<|end|>', '').replace('<|user|>', '').replace('<|system|>', '').replace('<|assistant|>', '')
        #MUST BE '¿Cuál es'
        final_ai_response = final_ai_response.partition('<|end|>')[0].strip().replace('\n\n', '\n').replace('<|end|>', '').replace('<|user|>', '').replace('<|system|>', '').replace('<|assistant|>', '')
        new_final_ai_response = final_ai_response.split('Unhelpful Answer:')[0].strip()
        new_final_ai_response = new_final_ai_response.split('Note:')[0].strip()
        new_final_ai_response = new_final_ai_response.split('Please provide feedback on how to improve the chatbot.')[0].strip()                 

        print("Final AI Response:")
        print(new_final_ai_response)
        
        st.write("AI Response:")
        st.write(new_final_ai_response)
