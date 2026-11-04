"""
LangChain Runnable Cookbook
============================

Runnable examples covered:
1. RunnableLambda
2. RunnablePassthrough
3. RunnableParallel
4. RunnableSequence
5. RunnableBranch
6. RunnablePassthrough.assign()
7. RunnableConfig (tags, metadata, run_name)
8. invoke()
9. batch()
10. stream()
11. ainvoke()

Install:
    pip install -U langchain-core

Run:
    python langchain_runnables_examples.py

These examples use Python functions only. No LLM, Azure endpoint, or API key
is required. Later, you can replace a RunnableLambda with an LLM or another
Runnable and compose them using the same LCEL patterns.
"""

import asyncio
from typing import Any

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. RunnableLambda: wrap an ordinary Python function as a Runnable
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    return text.strip().lower()


cleaner = RunnableLambda(clean_text)


def example_runnable_lambda() -> None:
    section("1. RunnableLambda")
    result = cleaner.invoke("   HELLO LANGCHAIN   ")
    print(result)  # hello langchain


# ---------------------------------------------------------------------------
# 2. RunnablePassthrough: pass the input through unchanged
# ---------------------------------------------------------------------------

def example_runnable_passthrough() -> None:
    section("2. RunnablePassthrough")
    passthrough = RunnablePassthrough()

    original_input = {
        "name": "Alex",
        "technology": ".NET",
    }

    result = passthrough.invoke(original_input)
    print(result)  # Same dictionary as the input


# ---------------------------------------------------------------------------
# 3. RunnableParallel: send the same input to multiple branches
# ---------------------------------------------------------------------------

def make_summary(data: dict[str, Any]) -> str:
    return f"Summary: {data['topic']} is a software development topic."


def make_keywords(data: dict[str, Any]) -> list[str]:
    return [data["topic"], "software", "development"]


def example_runnable_parallel() -> None:
    section("3. RunnableParallel")

    parallel = RunnableParallel(
        summary=RunnableLambda(make_summary),
        keywords=RunnableLambda(make_keywords),
    )

    result = parallel.invoke({"topic": "LangChain"})
    print(result)
    # {
    #   'summary': 'Summary: LangChain is a software development topic.',
    #   'keywords': ['LangChain', 'software', 'development']
    # }


# ---------------------------------------------------------------------------
# 4. RunnableSequence: execute each Runnable in order
# ---------------------------------------------------------------------------

def add_topic_key(text: str) -> dict[str, str]:
    return {"topic": text}


def format_explanation(data: dict[str, str]) -> str:
    return f"Here is a simple explanation of {data['topic']}."


def example_runnable_sequence() -> None:
    section("4. RunnableSequence")

    sequence = RunnableSequence(
        RunnableLambda(clean_text),
        RunnableLambda(add_topic_key),
        RunnableLambda(format_explanation),
    )

    result = sequence.invoke("   DEPENDENCY INJECTION   ")
    print(result)
    # Here is a simple explanation of dependency injection.


# ---------------------------------------------------------------------------
# 5. RunnableBranch: choose a Runnable based on a condition
# ---------------------------------------------------------------------------

def technical_answer(data: dict[str, str]) -> str:
    return f"Technical explanation for: {data['question']}"


def general_answer(data: dict[str, str]) -> str:
    return f"Simple explanation for: {data['question']}"


def example_runnable_branch() -> None:
    section("5. RunnableBranch")

    branch = RunnableBranch(
        (
            lambda data: any(
                word in data["question"].lower()
                for word in ["api", "code", "programming", ".net"]
            ),
            RunnableLambda(technical_answer),
        ),
        RunnableLambda(general_answer),  # Default branch
    )

    technical_result = branch.invoke(
        {"question": "How do I create a REST API in .NET?"}
    )
    general_result = branch.invoke(
        {"question": "What is photosynthesis?"}
    )

    print("Technical:", technical_result)
    print("General:", general_result)


# ---------------------------------------------------------------------------
# 6. RunnablePassthrough.assign(): preserve input and add computed fields
# ---------------------------------------------------------------------------

def calculate_word_count(data: dict[str, Any]) -> int:
    return len(data["question"].split())


def classify_question(data: dict[str, Any]) -> str:
    question = data["question"].lower()
    return "technical" if "api" in question or "code" in question else "general"


def example_passthrough_assign() -> None:
    section("6. RunnablePassthrough.assign()")

    enrich_input = RunnablePassthrough.assign(
        word_count=RunnableLambda(calculate_word_count),
        category=RunnableLambda(classify_question),
    )

    result = enrich_input.invoke(
        {"question": "How do I build an API?"}
    )

    print(result)
    # Original question is preserved, and word_count/category are added.


# ---------------------------------------------------------------------------
# 7. RunnableConfig: pass execution configuration
# ---------------------------------------------------------------------------

def example_runnable_config() -> None:
    section("7. RunnableConfig")

    runnable = RunnableLambda(clean_text)

    config = {
        "run_name": "clean_user_input",
        "tags": ["learning", "runnable-example"],
        "metadata": {
            "lesson": "LangChain Runnables",
            "environment": "development",
        },
    }

    result = runnable.invoke("  HELLO CONFIG  ", config=config)
    print("Result:", result)
    print("Config supplies execution metadata; it does not change the function's logic.")


# ---------------------------------------------------------------------------
# 8. invoke(): execute one input
# ---------------------------------------------------------------------------

def example_invoke() -> None:
    section("8. invoke()")

    runnable = RunnableLambda(lambda number: number * 2)
    result = runnable.invoke(10)

    print(result)  # 20


# ---------------------------------------------------------------------------
# 9. batch(): execute multiple inputs
# ---------------------------------------------------------------------------

def example_batch() -> None:
    section("9. batch()")

    runnable = RunnableLambda(lambda number: number * 2)
    results = runnable.batch([1, 2, 3, 4])

    print(results)  # [2, 4, 6, 8]


# ---------------------------------------------------------------------------
# 10. stream(): consume output chunks
# ---------------------------------------------------------------------------

def example_stream() -> None:
    section("10. stream()")

    # A simple RunnableLambda produces one output chunk.
    # LLMs that support streaming can produce multiple chunks/tokens.
    runnable = RunnableLambda(
        lambda text: f"Processed: {text}"
    )

    for chunk in runnable.stream("LangChain"):
        print("Chunk:", chunk)


# ---------------------------------------------------------------------------
# 11. ainvoke(): execute a Runnable asynchronously
# ---------------------------------------------------------------------------

async def example_ainvoke() -> None:
    section("11. ainvoke()")

    async def async_process(text: str) -> str:
        # In real applications, await async I/O here, e.g. an HTTP request.
        return f"Async result: {text.upper()}"

    runnable = RunnableLambda(async_process)

    result = await runnable.ainvoke("langchain")
    print(result)


def main() -> None:
    example_runnable_lambda()
    example_runnable_passthrough()
    example_runnable_parallel()
    example_runnable_sequence()
    example_runnable_branch()
    example_passthrough_assign()
    example_runnable_config()
    example_invoke()
    example_batch()
    example_stream()
    asyncio.run(example_ainvoke())

    section("Finished")
    print("All Runnable examples completed.")
    print("Remember: Runnables can be composed using the LCEL pipe operator |.")


if __name__ == "__main__":
    main()
