from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import StdOutCallbackHandler
from src.agents.agents import run_reader_agent
def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
         "messages" : [HumanMessage(content=f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content

    print("\n search result ",state['search_results'])


    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)


    state["scraped_content"] = run_reader_agent(
    topic,
    state["search_results"]
)

    print("\nSTEP 2 COMPLETED: All research sources processed.")

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"""
                        Research Topic: {topic}

                        Search Results:
                        {state["search_results"]}

                        Collect relevant research information from
                        the URLs using your available tools.
                        """
                    )
                ]
            },
            config={"callbacks": [StdOutCallbackHandler()]}
        )

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nScraped content:\n", state["scraped_content"])


    #step 3 - writer chain 

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])


    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
    "topic": topic,
    "research": research_combined,
    "report": state["report"]
})

    print("\n critic report \n", state['feedback'])

    return state