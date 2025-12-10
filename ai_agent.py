import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
dotenv.load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


model = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

from typing import Dict, TypedDict , Union
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph , START , END

class AgentState(TypedDict):
    messages: list[Union[AIMessage, HumanMessage]]

def process(state:AgentState) -> AgentState:
    response = model.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START , "process")
graph.add_edge("process" , END)
app = graph.compile()

history = []

user_input = input("User: ")
while user_input.lower() != "exit":
    history.append(HumanMessage(content=user_input))
    state = {
        "messages": history
    }
    result = app.invoke(state)
    ai_message = result['messages'][-1]
    print(f"AI: {ai_message.content}")
    history.append(ai_message)
    user_input = input("User: ")