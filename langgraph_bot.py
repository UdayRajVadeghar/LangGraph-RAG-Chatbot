import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END

load_dotenv()

# Define state structure
class State(TypedDict):
    input: str
    output: str

# Load Vector DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever()

# Tool to use for question-answering
tool = create_retriever_tool(
    retriever,
    name="uday_context",
    description="Ask about Uday Raj – his work, projects, profile"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")



# Define a node
def retrieve_answer(state: State) -> State:
    query = state["input"]  
    context = tool.run(query)

    # Use the LLM to generate a response based on the retrieved context
    prompt = f"""Based on the following context about Uday Raj, answer the user's question.

Context: {context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return {"input": query, "output": response.content}

# LangGraph setup
graph = StateGraph(State)
graph.add_node("ask_uday", retrieve_answer)
graph.set_entry_point("ask_uday")
graph.add_edge("ask_uday", END)
compiled = graph.compile()

# Generate visual diagram of the graph
try:
    png_data = compiled.get_graph().draw_mermaid_png()
    with open("./graph_diagram.png", "wb") as f:
        f.write(png_data)
    print("📊 Graph diagram saved as 'graph_diagram.png'")
except Exception as e:
    print(f"📊 Could not create PNG diagram: {e}")
    print("   Graph Structure: ask_uday → END")
    print("   (Retrieves context and generates response)")

print()

# CLI loop
print("🤖 Uday Bot is ready! Ask me anything about Uday Raj.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    q = input("Ask about Uday → ")
    if q.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    try:
        result = compiled.invoke({"input": q})
        print(f"\n🤖 Uday Bot: {result['output']}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
