FINAL SUBMISSION — MEDICATION PERSISTENCE

Contents:
- solution_document.docx: complete technical solution document.
- solution_document.md: Markdown companion.
- presentation_beautiful_ai.pdf: the 8-slide PDF supplied by the user, included unchanged.
- project/: complete source code, synthetic data, dashboard, RAG assistant, tests and evaluation outputs.

Run:
1. cd project
2. python -m venv .venv
3. Activate the virtual environment.
4. python -m pip install -r requirements.txt
5. python run_project.py
6. Open dashboard/index.html in a browser.
7. python -m pip install -r rag_assistant/requirements.txt
8. python -m unittest tests.test_analysis -v
9. python -m unittest rag_assistant.tests.test_rag -v
10. uvicorn rag_assistant.api:app --reload
11. Optional LLM: configure OPENAI_API_KEY. Without it, extractive fallback is used.

All datasets are synthetic. See solution_document.docx for assumptions, caveats, metrics, security and deployment considerations.
