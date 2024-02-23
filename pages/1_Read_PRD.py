import streamlit as st
import docx 
import os
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain_core.documents.base import Document
from langchain.globals import set_debug
import  tiktoken

def calculate_tokens(text):
    return len(tiktoken.encoding_for_model('gpt-3.5-turbo').encode(text))

def calculate_tokens_for_document(document):
    all_text = ""
    for para in document.paragraphs:
        all_text += para.text
        all_text += "\n"

    return calculate_tokens(all_text)

def load_document(prd_file):
    """Saves the Word document uploaded by the user to a temp location, 
       parses it using python-docx, deletes the temp file, and returns 
       the name of the file and the parsed Word document.
    """
    prd_bytes = prd_file.getvalue()
    os.makedirs("temp", exist_ok=True)
    temp_file_name = os.path.join("temp", prd_file.name)
    with open(temp_file_name, 'wb') as f:
        f.write(prd_bytes)

    document = docx.Document(temp_file_name)
    os.remove(temp_file_name)
    return temp_file_name,document

def chunk_using_nltk(chunks):
    """Breaks the chunks down using Natural Language Toolkit.
    Using the punkt library."""
    import nltk
    nltk.download('punkt')

    from langchain.text_splitter import NLTKTextSplitter

    text_splitter = NLTKTextSplitter(chunk_size=2000)
    chunks = text_splitter.split_documents(chunks)
    return chunks
    
def parse_word_doc_single_chunk(temp_file_name, document):
    """Create a langchain Document from the text, 
       as a single chunk."""
    text = ""
    metadata = {"source": os.path.basename(temp_file_name)}
    for para in document.paragraphs:
        text += para.text
        text += "\n\n"
    return [Document(page_content=text, metadata=metadata)]

def documents_to_chroma(chunks, embeddings):
    """Save the chunks to Chroma as separate documents."""
    from langchain_community.vectorstores.chroma import Chroma
    ids = [str(i+1) for i in range(len(chunks))]
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, ids=ids)
    return vectorstore

@st.cache_data
def get_requirements_from_mapreduce_chain(_llm, _chunks):

    _chunks = chunk_using_nltk(_chunks)

    from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, LLMChain, StuffDocumentsChain
    from langchain.prompts import PromptTemplate

    map_template = """The following is a set of fragments from a product requirements document. 
                      Go through it and summarize it as a list of product requirement summaries. 
                      Use a single sentence for each requirement summary. Provide the response 
                      with each requirement on a different line:
                      ====== begin product requirements document fragments ======
                      {document_fragments}
                      ====== end product requirements document fragments ======
                      List of requirements:
                      """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=_llm, prompt=map_prompt)


    reduce_template = """The following contains multiple requirements, with one requirement on each line. 
                         ===== begin requirements  =====
                         {requirements_list}
                         ===== end requirements  =====
                         Take these and identify any duplicate or redundant requirements.
                         Eliminate the redundant and duplicate requirements, and create a new 
                         single list of requirements. Provide the response in a JSON object as a list of strings.
                         Format Instructions:
                         Please provide the output as a JSON using the schema {{ "requirements": [ "requirement1", "requirement2" ] }}
                         List of requirements:
                        """
    reduce_prompt = PromptTemplate(template=reduce_template, 
                                   input_variables=["requirements_list"]
                                   )
    reduce_chain = LLMChain(llm=_llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="requirements_list")

    reduce_requirements_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max = 4000
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_requirements_chain,
        document_variable_name="document_fragments",
        return_intermediate_steps=False
    )

    requirements = map_reduce_chain.run(_chunks)

    import json
    return json.loads(requirements)["requirements"]

from llmfactory import get_gpt35_llm, get_gpt35turbo

def get_stories(selected_reqs, vectorstore):
    import json

    stories = []
    for req in selected_reqs:
        response = get_story(req, vectorstore.as_retriever())
        stories.append((response["query"], json.loads(response["result"])))
    return stories

@st.cache_data
def get_story(req, _retriever):
    expand_llm = get_gpt35turbo()

    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    prompt_template = PromptTemplate(
        template="""You are a Product Manager for an Agile program. You create stories for 
                    implementation by the engineering team, that follow the INVEST framework,
                    where the letters stand for
                    'I' ndependent (of all others)
                    'N' egotiable (not a specific contract for features)
                    'V' aluable (or vertical)
                    'E' stimable (to a good approximation)
                    'S' mall (so as to fit within an iteration)
                    'T' estable (in principle, even if there isn't a test for it yet).
                    Please create a user story for the following requirement:
                    The context for the requirements is given below.
                    ==== begin context ====
                    {context}
                    ==== end context ====
                    ==== begin requirement ====
                    {question}
                    ==== end requirement ====
                    
                    Each story must have a 
                    1. Title
                    2. Description
                    3. List of Acceptance Criteria
                    Provide the output as a JSON using the following schema:
                    {{ "title": "Story Title", "description": "Story Description", "acceptance_criteria": [ "criteria 1", "criteria 2", "criteria 3" ] }}
                    """,
        input_variables=['requirement'])

    chain = RetrievalQA.from_chain_type(
        llm=expand_llm,
        retriever=_retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return chain({"query": req})

st.header("Time to import the PRD")

prd_file = st.file_uploader("Upload a PRD document", ["docx"])

if prd_file is not None:
    set_debug(True)
    # File has been uploaded. Parse it
    temp_file_name, document = load_document(prd_file)

    chunks = parse_word_doc_single_chunk(temp_file_name, document)

    from langchain_openai import OpenAIEmbeddings
    vectorstore = documents_to_chroma(chunks, OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    requirements = get_requirements_from_mapreduce_chain(get_gpt35_llm(), chunks)
    for i, requirement in enumerate(requirements):
        st.markdown(f"{i+1}. {requirement}")

    selected_reqs = st.multiselect("Select the requirements to expand", requirements, [])

    stories = get_stories(selected_reqs, vectorstore)

    for requirement, story in stories:
        st.subheader(requirement)
        st.markdown(f"**Title**: {story['title']}")
        st.markdown(f"**Description**: {story['description']}")
        st.markdown(f"**Acceptance Criteria**")
        for i, criteria in enumerate(story["acceptance_criteria"]):
            st.markdown(f"{i+1}. {criteria}")
