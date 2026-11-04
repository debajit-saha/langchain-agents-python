
import os
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 1. Connect to Azure AI Foundry
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

model = ChatOpenAI(
    base_url=endpoint,
    api_key=api_key,
    model=deployment_name
)

# 2. Define tools


@tool
def get_order_status(order_id: str) -> str:
    """Get the current status of an order."""

    orders = {
        "ORD-001": "Dispatched",
        "ORD-002": "Processing",
        "ORD-003": "Delivered"
    }

    return orders.get(order_id, "Order not found")


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an existing customer order."""

    # Simulated cancellation
    return f"Order {order_id} has been cancelled successfully."


# 3. Create agent with HITL middleware
agent = create_agent(
    model=model,
    tools=[get_order_status, cancel_order],
    system_prompt="""
    You are a helpful order management assistant.

    Use tools to check order status.
    Cancel orders only when requested by the user.
    """,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "cancel_order": True
            }
        )
    ],
    checkpointer=InMemorySaver()
)

# 4. Define conversation thread
config = {
    "configurable": {
        "thread_id":  str(uuid.uuid4())
    }
}

# 5. Invoke agent
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Cancel order ORD-002"
            }
        ]
    },
    config=config
)

# 6. Check if human approval is required

state = agent.get_state(config)

if state.interrupts:

    print("Human approval required!")

    for interrupt in state.interrupts:
        print("Approval details:", interrupt.value)

    decision = input("Approve cancellation? (yes/no): ")

    if decision.lower() == "yes":
        decision_value = {"type": "approve"}
    else:
        decision_value = {
            "type": "reject",
            "message": "Cancellation rejected by administrator."
        }

    # Resume interrupted execution
    result = agent.invoke(
        Command(
            resume={
                "decisions": [decision_value]
            }
        ),
        config=config
    )

# 7. Print final response

print("\nFinal response:")

for message in result["messages"]:
    if message.type == "ai" and message.content:
        print(message.content)
