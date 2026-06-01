"""
This contains a simple script to the pipeline to retrieve the answer
"""

from .db_retriever import DBRetriever
from .qa_inference import QA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from src.prompt import prompt_template
import torch


def pipeline(
    api_key: str,
    question: str,
    hf_auth: str,
    index_name="smartpub",
    model_name="meta-llama/Llama-2-13b-chat-hf",
    device=torch.device("cpu"),
    verbose=True,
    batch_size=32,
    k=10,
) -> str:
    """
    Create a pipeline for the question answering
    :param api_key: The API key for accessing the Pinecone service.
    :param question: question as the query
    :param str hf_auth: The HuggingFace authentication token
    :param device: device to run on (auto-detected at runtime)
    :return: final answer as output
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Pinecone retriever
    retriever = DBRetriever(
        api_key=api_key,
        hf_auth=hf_auth,
        index_name=index_name,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )

    # Load LLM
    qa = QA(prompt=question, device=device, hf_auth=hf_auth)
    qa.qa_inference(qa.task, qa.model_name)

    # Build LCEL RAG chain
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "input"]
    )
    combine_chain = create_stuff_documents_chain(qa.llm, prompt)
    rag_chain = create_retrieval_chain(
        retriever.vectorstore_db.as_retriever(search_kwargs={"k": k}),
        combine_chain,
    )

    result = rag_chain.invoke({"input": question})
    return result["answer"]
