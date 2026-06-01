"""
This contains a simple script to set up the connection to the Pinecone database that contains all of the documents
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from src.prompt import prompt_template
import pinecone
import torch
from transformers import LlamaTokenizer
from .qa_inference import QA

class DBRetriever:

    def __init__(self, api_key:str, hf_auth:str, index_name:str="smartpub", 
                    model_name:str='sentence-transformers/all-MiniLM-L6-v2', batch_size=32, device=torch.device('cpu')):

        """
        This class uses a vectorstore as a retriever to be part of the QA pipeline to retrieve the most relevant documents
        Default model is the SentenceTransformer Mini Llama

        :param str api_key: The API key for accessing the service.
        :param str hf_auth: The HF authentication key to retrieve
        :param str index_name: The name of the index to be used (default is "smartpub").
        :param str model_name: The name of the pre-trained model to be used (default is 'sentence-transformers/all-MiniLM-L6-v2').
        :param int batch_size: The batch size for processing data (default is 32).
        :param torch.device device: The device to be used for processing (default is for CPU).

        
        """

        # Initialize Pinecone client
        pc = pinecone.Pinecone(api_key=api_key) 

        # Initialize the existing index
        self.index = pc.Index(name=index_name)


        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf",token=hf_auth)    
        self.embed_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': batch_size}
            )
        self.vectorstore_db = PineconeVectorStore(index=self.index, embedding=self.embed_model, text_key='relations')

        self.device = device
        self.hf_auth = hf_auth
        

    def run(self, question, k=10):
        """Create a pipeline for the question answering

            :param question: question as the query
            :param k: number of documents to retrieve
            :return: final answer as output
        """

        qa = QA(prompt=question, device=self.device, hf_auth=self.hf_auth)
        qa.qa_inference(qa.task, qa.model_name)

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
        combine_chain = create_stuff_documents_chain(qa.llm, prompt)
        rag_chain = create_retrieval_chain(
            self.vectorstore_db.as_retriever(search_kwargs={"k": k}),
            combine_chain,
        )

        result = rag_chain.invoke({"input": question})
        return result["answer"]
