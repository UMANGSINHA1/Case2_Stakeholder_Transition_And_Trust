"""Interactive terminal chat: python -m rag_assistant.cli"""
from .rag_engine import answer_question

print("Medication Persistence Methodology Assistant (type 'exit' to quit)")
while True:
    try:
        question = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye")
        break
    if question.lower() in {"exit", "quit"}:
        break
    if len(question) < 3:
        print("Please enter a more specific question.")
        continue
    result = answer_question(question)
    print("\nAssistant:", result["answer"])
    print(f"\nMode: {result['mode']} | Abstained: {result['abstained']}")
    if result["citations"]:
        print("Sources:", ", ".join(result["citations"]))
