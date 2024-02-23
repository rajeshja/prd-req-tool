def get_gpt35_llm():
    return get_gpt35turbo()

def get_gpt35turbo():
    """This returns the ChatGPT 3.5 Turbo Instruct model. The context window is 4k.
    This model is an Instruct model, and is best suited to for NLP tasks and when
    you want it to stick to your desired output.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
def get_gpt35_instruct():
    """This returns the ChatGPT 3.5 Turbo Instruct model. The context window is 4k.
    This model is an Instruct model, and is best suited to for NLP tasks and when
    you want it to stick to your desired output.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

def get_gpt35_16k():
    """This returns the ChatGPT 3.5 Turbo model. The context window is 16k.
    However, this model is not an Instruct model, and is best suited to Chat mode.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

def get_gpt4turbo_128k():
    """This returns the ChatGPT 4 Turbo model. The context window is 128k.
    The model is currently in preview, and is cheaper than standard gpt4 
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
