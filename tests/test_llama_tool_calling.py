"""
Simple test to verify Llama 3.1 8B tool calling works.
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Define a simple test tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny, 72°F"

# Create LLM
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

print("Creating agent with tool calling...")
# Create agent with tool
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
)

print("Invoking agent...")
# Test tool calling
result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in San Francisco?")]
})

print("\n=== RESULT ===")
print(f"Messages: {len(result['messages'])}")
for i, msg in enumerate(result['messages']):
    print(f"\n[{i}] {type(msg).__name__}: {msg.content[:200]}")

print("\n✓ Tool calling works!")
