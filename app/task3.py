import asyncio
import json
from dataclasses import dataclass, field


ORDERS = {
    "1001": {
        "order_id": "1001",
        "status": "shipped",
        "eta": "2026-11-05"
    },
    "1002": {
        "order_id": "1002",
        "status": "processing",
        "eta": "not available yet"
    }
}

CUSTOMERS = {
    "C001": {
        "customer_id": "C001",
        "name": "Arun",
        "tier": "gold"
    },
    "C002": {
        "customer_id": "C002",
        "name": "Priya",
        "tier": "standard"
    }
}


def get_order_status(order_id: str) -> dict:
    return ORDERS.get(
        order_id,
        {"error": f"Order {order_id} not found"}
    )


def get_customer(customer_id: str) -> dict:
    return CUSTOMERS.get(
        customer_id,
        {"error": f"Customer {customer_id} not found"}
    )


def calculate_discount(amount: float, percent: float) -> dict:
    if amount < 0 or not 0 <= percent <= 100:
        return {
            "error": "Amount must be >= 0 and percent must be 0 to 100"
        }

    discount = round(amount * percent / 100, 2)

    return {
        "amount": amount,
        "percent": percent,
        "discount": discount,
        "final_amount": round(amount - discount, 2)
    }


TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "get_customer": get_customer,
    "calculate_discount": calculate_discount
}


TOOL_SCHEMAS = [
    {
        "name": "get_order_status",
        "description": "Get the status and estimated delivery date for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"],
            "additionalProperties": False
        }
    },
    {
        "name": "get_customer",
        "description": "Get a customer's details using their customer ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"}
            },
            "required": ["customer_id"],
            "additionalProperties": False
        }
    },
    {
        "name": "calculate_discount",
        "description": "Calculate a discount and final price.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "percent": {"type": "number"}
            },
            "required": ["amount", "percent"],
            "additionalProperties": False
        }
    }
]


def execute_tool(name: str, arguments: dict) -> dict:
    function = TOOL_FUNCTIONS.get(name)

    if function is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        return function(**arguments)
    except (TypeError, ValueError) as error:
        return {"error": f"Invalid arguments: {error}"}


async def execute_tool_round(calls: list[dict]) -> list[dict]:
    async def run(call):
        result = await asyncio.to_thread(
            execute_tool,
            call.get("name", ""),
            call.get("arguments", {})
        )

        return {
            "tool_call_id": call.get("tool_call_id"),
            "name": call.get("name"),
            "result": result
        }

    return await asyncio.gather(*(run(call) for call in calls))


@dataclass
class Conversation:
    messages: list[dict] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content
        })

    def full_history(self) -> list[dict]:
        history = list(self.messages)

        if self.summary:
            history.insert(0, {
                "role": "system",
                "content": "Summary of earlier conversation: " + self.summary
            })

        return history

    def trim_to_budget(self, history_budget: int = 3000) -> dict:
        def estimate(text):
            return max(1, len(text.split()) * 4 // 3)

        before = sum(
            estimate(message["content"])
            for message in self.messages
        )

        kept = []
        used = 0

        for message in reversed(self.messages):
            size = estimate(message["content"])

            if used + size > history_budget:
                break

            kept.append(message)
            used += size

        kept.reverse()

        removed_count = len(self.messages) - len(kept)
        removed = self.messages[:removed_count]

        if removed:
            old_text = " ".join(
                message["content"] for message in removed
            )

            self.summary = (
                self.summary + " " + old_text
            ).strip()[-3000:]

        self.messages = kept

        after = sum(
            estimate(message["content"])
            for message in self.messages
        )

        return {
            "before_tokens": before,
            "after_tokens": after,
            "system_budget": 500,
            "history_budget": history_budget,
            "output_budget": 1000,
            "summary": self.summary
        }


def demonstrate_error_cases():
    return [
        execute_tool("get_order_status", {"order_id": "9999"}),
        execute_tool("get_customer", {"customer_id": "missing"}),
        execute_tool(
            "calculate_discount",
            {"amount": -5, "percent": 10}
        ),
        execute_tool("unknown_tool", {})
    ]


def run_openai_tool_loop(user_prompt: str, max_rounds: int = 5) -> dict:
    from app.llm import get_openai_client
    from app.config import settings

    client = get_openai_client()

    model = (
        settings.OPENROUTER_MODEL
        if settings.OPENROUTER_API_KEY
        else settings.OPENAI_MODEL
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        }
        for tool in TOOL_SCHEMAS
    ]

    messages = [{"role": "user", "content": user_prompt}]

    for round_number in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            return {
                "answer": assistant_message.content or "",
                "rounds": round_number,
                "messages": messages
            }

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                }
                for call in assistant_message.tool_calls
            ]
        })

        calls = []

        for call in assistant_message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                name = call.function.name
            except (json.JSONDecodeError, TypeError):
                arguments = {}
                name = "__invalid_arguments__"

            calls.append({
                "tool_call_id": call.id,
                "name": name,
                "arguments": arguments
            })

        results = asyncio.run(execute_tool_round(calls))

        for result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": json.dumps(result["result"])
            })

    return {
        "answer": "Maximum tool rounds reached.",
        "rounds": max_rounds,
        "messages": messages
    }


if __name__ == "__main__":
    print("Tool examples:")
    print(get_order_status("1001"))
    print(get_customer("C001"))
    print(calculate_discount(1000, 10))

    print("\nError demonstrations:")
    print(demonstrate_error_cases())

    conversation = Conversation()
    conversation.add("user", "My favourite colour is blue.")
    conversation.add("assistant", "I will keep that in this conversation.")

    print("\nConversation history:")
    print(conversation.full_history())