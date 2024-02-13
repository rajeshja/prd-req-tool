import streamlit as st
import docx 
import os
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain_core.documents.base import Document
import  tiktoken

st.header("Time to import the PRD")

prd_file = st.file_uploader("Upload a PRD document", ["pdf", "docx"])

def calculate_tokens(chunk):
    return len(tiktoken.encoding_for_model('gpt-3.5-turbo').encode(chunk.page_content))


if prd_file is not None:
    prd_bytes = prd_file.getvalue()
    os.makedirs("temp", exist_ok=True)
    temp_file_name = os.path.join("temp", prd_file.name)
    with open(temp_file_name, 'wb') as f:
        f.write(prd_bytes)

    chunks = []

    current_fragment = ""
    current_metadata = {"source": temp_file_name}
    prev_para_type = ""

    document = docx.Document(temp_file_name)
    for para in document.paragraphs:
        if para.style.name.startswith("Heading") and prev_para_type != "heading":
            # is header 1
            if current_fragment != "":
                chunk = Document(page_content=current_fragment, metadata=current_metadata)
                chunks.append(chunk)
            current_fragment = para.text 
            current_metadata = {"source": os.path.basename(temp_file_name), "heading": para.text}
            prev_para_type = "heading"
        else:
            current_fragment += f"\n{para.text}"
            prev_para_type = "para"

    if current_fragment != "":
        chunk = Document(page_content=current_fragment, metadata=current_metadata)
        chunks.append(chunk)

    st.write(f"Length {len(chunks)}")

    import nltk
    nltk.download('punkt')

    from langchain.text_splitter import NLTKTextSplitter

    text_splitter = NLTKTextSplitter(chunk_size=1000)
    chunks = text_splitter.split_documents(chunks)

    st.write(f"Length {len(chunks)}")

    for chunk in chunks:
        st.write(f"Bytes: {len(chunk.page_content)}, Tokens: {calculate_tokens(chunk)}")
        st.subheader("Metadata")
        st.json(chunk.metadata)
        st.subheader("Document content")
        st.write(chunk.page_content)
        st.divider()

    from langchain_community.vectorstores.chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain import hub
    from langchain.chains import RetrievalQA, MapReduceDocumentsChain, ReduceDocumentsChain

    map_template = """
        The following """

    func_req_query = """Look at the given context, and see if this text contains any functional requirements.
                        If there are any functional requirements mentioned, give a short summary of the requirement.
                        Also provide a short title of the requirement. the title should be within 4 words.
                        Do not include non-functional requirements in your response. 
                        If there are no functional requirements, use the title 'none'.
                        Provide the answer in the following structure: \{ 'title': 'Requirement Title', 'summary': 'Requirement Summary' \}"""

    ids = [str(i+1) for i in range(len(chunks))]
    vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(), ids=ids)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate

    prompt_template = hub.pull("rlm/rag-prompt")
    print("From hub:")
    print(prompt_template)

    # """
    #     input_variables=['context', 'question'] 
    #     """
    prompt_template = PromptTemplate(input_variables=["context", "question"], 
                                     template="""You are the assistant to a Product Manager for question-answering tasks.
                                                 Use the following pieces of retrieved context to answer the question.
                                                 If you don't know the answer, just say that you don't know.
                                                 Use three sentences maximum and keep the answer concise.
                                                 Question: {question} 
                                                 Context: {context} 
                                                 Answer:""")
    
    print("Custom prompt:")
    print(prompt_template)

    # chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever = vectorstore.as_retriever(),
    #     return_source_documents=False,
    #     chain_type_kwargs={"prompt": prompt_template}
    # )

    embedding_vector = OpenAIEmbeddings().embed_query(func_req_query)
    docs_retrieved = vectorstore.similarity_search_by_vector_with_relevance_scores(embedding_vector, k=50)

    st.header("Docs from Chroma")

    # docs_retrieved = retriever.invoke(func_req_query)

    for i, (doc, relevance) in enumerate(docs_retrieved):
        st.write(f"Sr No: {i+1}, Relevance: {relevance}, Tokens: {calculate_tokens(doc)}")
        st.write(doc)
        st.divider()

    from string import Template

    prompt = Template("""You are the assistant to a Product Manager for question-answering tasks.
                                                 Use the following pieces of retrieved context to answer the question.
                                                 If you don't know the answer, just say that you don't know.
                                                 Use three sentences maximum and keep the answer concise.
                                                 Question: $question
                                                 Context: $context
                                                 Answer:""")

    for i in range(2, 4):
        # question = func_req_query
        # context = docs_retrieved[i]
        # response = chain.invoke({"query": func_req_query, "context": docs_retrieved[i]})
        st.divider()
        st.write(llm(prompt.substitute(question=func_req_query, context=docs_retrieved[i])))
        # st.write(response["result"])
        # print(response["result"])
        st.divider()
        

    # chain.invoke({"query": func_req_query})
    # results = vectorstore.q

    # loader = Docx2txtLoader(temp_file_name)
    # chunks = loader.load()

