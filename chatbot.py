from langgraph.graph import StateGraph, MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from tools import query_knowledge_base, search_for_product_reccommendations, data_protection_check, create_new_customer, place_order, retrieve_existing_customer_orders,send_order_confirmation_email
from dotenv import load_dotenv
import os
load_dotenv()


prompt = """#Purpose

You are a customer service chatbot for an online elastic products store. You can help customers achieve the goals listed below.

#Goals

1.Answer questions users might have relating to the services offered.
2.Recommend elastic products to users based on their preferences.
3.Help customers check on an existing order or place a new order.
4.To place and manage orders, you will need a customer profile (with an associated ID). If the customer already has a profile, perform a data protection check to retrieve their details. If not, create them a profile.


#Tone

Helpful and friendly. Use Gen-Z emojis to keep things lighthearted.
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        ('system', prompt),
        ('placeholder', "{messages}")
    ]
)


tools = [query_knowledge_base, search_for_product_reccommendations, data_protection_check, create_new_customer, place_order, retrieve_existing_customer_orders,send_order_confirmation_email]

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     openai_api_key=os.environ['OPENAI_API_KEY']
# )
llm=ChatGroq(model="llama-3.1-70b-versatile",api_key=os.environ['GROQ_API_KEY'])

llm_with_prompt = chat_template | llm.bind_tools(tools)


def call_agent(message_state: MessagesState):
    
    response = llm_with_prompt.invoke(message_state)

    return {
        'messages': [response]
    }

def is_there_tool_calls(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tool_node'
    else:
        return '__end__'


graph = StateGraph(MessagesState)

tool_node = ToolNode(tools)

graph.add_node('agent', call_agent)
graph.add_node('tool_node', tool_node)

graph.add_conditional_edges(
    "agent",
    is_there_tool_calls
)
graph.add_edge('tool_node', 'agent')

graph.set_entry_point('agent')

app = graph.compile()

