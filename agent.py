from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

GROQ_MODEL = "openai/gpt-oss-120b"


@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input is a short search query.
    Returns titles, links and snippets of the top results."""
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:
        return f"Search failed: {e}. Try a different query."

    if not results:
        return "No results found. Try a different query."

    lines = []
    for r in results:
        lines.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n")
    return "\n".join(lines)


def run_research(topic: str, groq_api_key: str) -> str:
    # CrewAI strips ONE leading "openai/", so we write it twice.
    # Groq then receives the real ID: "openai/gpt-oss-120b".
    llm = LLM(
        model=f"openai/{GROQ_MODEL}",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        temperature=0.3,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research '{topic}' thoroughly and write a clear, accurate report.",
        backstory=(
            "You are an expert researcher. You search the web several times with "
            "different queries, compare sources, and never invent facts or links."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        max_iter=8,            # prevents endless loops
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=(
            f"Research the topic: {topic}\n\n"
            "Use the search tool 3-5 times with different queries. "
            "Then write a well-structured report."
        ),
        expected_output=(
            "A markdown report with: a title, a short summary, 3-5 key sections "
            "with headings, a conclusion, and a 'Sources' list with URLs."
        ),
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False)
    return crew.kickoff().raw
