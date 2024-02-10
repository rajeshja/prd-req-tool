import streamlit as st
import docx 
import os
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain_core.documents.base import Document
import  tiktoken

st.header("Time to import the PRD")

prd_file = st.file_uploader("Upload a PRD document", ["pdf", "docx"])

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
        st.write(f"Bytes: {len(chunk.page_content)}, Tokens: {len(tiktoken.encoding_for_model('gpt-3.5-turbo').encode(chunk.page_content))}")
        st.subheader("Metadata")
        st.json(chunk.metadata)
        st.subheader("Document content")
        st.write(chunk.page_content)
        st.divider()

    from langchain_community.vectorstores.chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    os.environ["OPENAI_API_KEY"] = ""

    ids = [str(i+1) for i in range(len(chunks))]
    vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(), ids=ids)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    st.write(retriever.invoke("Identify the Functional Requirements"))

    # results = vectorstore.q

    # loader = Docx2txtLoader(temp_file_name)
    # chunks = loader.load()

