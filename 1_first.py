'''
########## Installation ##########
pip install langchain pandas numpy 
pip install langchain-community langchain-openai
'''
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.schema.output_parser import StrOutputParser

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

# Initialize the Ollama model
llm = Ollama(model="gemma")  # Replace "llama2" with the Ollama model of your choice
output_parser = StrOutputParser()

# Chain the prompt and model
chain = prompt | llm | output_parser

# Ask the user for input
input_text = input("Enter your query: ")

# Generate and display the output
if input_text:
    result = chain.invoke({"question": input_text})
    print("Response:", result)
