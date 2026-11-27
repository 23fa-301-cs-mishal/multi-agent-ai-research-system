from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url,extract_pdf_text,split_into_chunks
from dotenv import load_dotenv
from urllib.parse import urlparse
import re
import time

load_dotenv()

# Model Initialization
llm = ChatGroq(model = "openai/gpt-oss-20b",temperature=0)


# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search],
       system_prompt="you are the search agent your responsibilty is to use the `web_search` tool on the given user query and return those 3 URLs with their titles.Return the URLs you recieve from running the tool dont invent your URLs"
    )

# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[
            scrape_url,
            
        ],
        system_prompt="""
        You are the Reader Agent.

        Your responsibilities are:

        1. Read the research topic and provided URLs to you.
        2. Use scrape_url for HTML webpages.
        3. Extract and analyze research information.
        4. Preserve source URLs, authors, titles and publication years.
        5. Return organized research findings.

        Do not invent information or any URL from your side.
        """
    )


def run_reader_agent(topic, search_results):

    # Extract URLs from search results
    urls = re.findall(r"https?://[^\s|)]+", search_results)

    all_findings = []

    # Process each research source
    for url in urls:

        url = url.rstrip(".,;]")

        print(f"\nProcessing source: {url}")

        try:

            # STEP 1: Extract complete text
            path = urlparse(url).path.lower()

            if path.endswith(".pdf"):

                text = extract_pdf_text(url)

            else:

                text = scrape_url.invoke({
                    "url": url
                })

            if not text or len(text.strip()) < 100:

                print("No readable content found.")
                continue

            # STEP 2: Divide text into chunks
            chunks = split_into_chunks(
                text,
                chunk_size=1500,
                overlap=150
            )

            print(f"Total chunks: {len(chunks)}")

            # STEP 3: Process 3 chunks per LLM call
            batch_size = 3

            total_batches = (
                len(chunks) + batch_size - 1
            ) // batch_size

            for i in range(0, len(chunks), batch_size):

                batch = chunks[i:i + batch_size]

                batch_number = (i // batch_size) + 1

                print(
                    f"Processing batch "
                    f"{batch_number}/{total_batches}"
                )

                # Combine the 3 chunks
                batch_text = ""

                for j, chunk in enumerate(batch):

                    batch_text += (
                        f"\n\nCHUNK {i + j + 1}:\n"
                        f"{chunk}"
                    )

                # LLM prompt
                prompt = f"""
You are a Research Reader Agent.

Research Topic: {topic}

Source URL: {url}

You are processing batch {batch_number}
out of {total_batches}.

Analyze ALL the chunks provided below.

Extract detailed research information:

1. Important research findings.
2. Research methodologies.
3. Algorithms and techniques.
4. Results and statistics.
5. Important conclusions.
6. Research gaps and limitations.

Instructions:

- Analyze all three chunks.
- Do not ignore relevant information.
- Preserve important technical details.
- Do not invent information.
- Keep the source URL attached to your findings.
- If information is repeated, avoid unnecessary repetition.

Research Text:

{batch_text}
"""

                try:

                    response = llm.invoke(prompt)

                    findings = response.content

                    all_findings.append(
                        f"""
SOURCE URL: {url}

BATCH: {batch_number}

RESEARCH FINDINGS:

{findings}
"""
                    )

                except Exception as e:

                    print(
                        f"Batch {batch_number} failed: {e}"
                    )

                    all_findings.append(
                        f"""
FAILED BATCH:
Source: {url}
Batch: {batch_number}
Error: {str(e)}
"""
                    )

                # Wait before the next Groq request
                if i + batch_size < len(chunks):
                    time.sleep(12)

        except Exception as e:

            print(f"Source processing failed: {e}")

    # STEP 4: Return all collected findings
    return "\n\n".join(all_findings)

#writer chain 


writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert academic research writer specializing in IEEE-style
review papers.

Your responsibility is to transform the provided research material
into a comprehensive, well-structured, factual, and professional
IEEE-style review paper.

Follow academic writing standards and maintain a logical flow
throughout the paper.

IMPORTANT WRITING RULES:

1. Use only the information provided in the research material.
2. Never invent authors, publication years, research findings,
   statistics, DOIs, URLs, or references.
3. Use formal academic language and avoid conversational expressions.
4. Write detailed paragraphs with clear explanations.
5. Avoid unnecessary repetition and irrelevant information.
6. Support factual statements with appropriate IEEE citations.
7. Use IEEE numbered in-text citations such as [1], [2], [3].
8. Assign reference numbers according to their first appearance.
9. Reuse the same reference number whenever citing the same source.
10. Every in-text citation must correspond to an entry in References.
11. Do not create references for sources that are not provided.
12. Clearly distinguish established research findings from
    your own synthesis and observations.
13. Do not fabricate comparisons, research gaps, or conclusions.
14. If information required for a section is unavailable,
    acknowledge the limitation instead of making assumptions.
15. Maintain consistency in terminology and technical explanations.
16. Use tables where they genuinely improve comparison.
17. Do not include the same information repeatedly in different sections.
18. Keep the report comprehensive but concise enough to avoid
    unnecessary token consumption.

"""),

    ("human", """
Write a comprehensive IEEE-style review paper on the following topic.

TOPIC:
{topic}

RESEARCH GATHERED:
{research}

Follow the complete structure below.

TITLE

Create a clear, specific, and academically appropriate title
for the review paper.

ABSTRACT

Write an abstract of approximately 150–250 words covering:

- Background of the research topic.
- Purpose and scope of the review.
- Research approaches and techniques discussed.
- Main findings from the reviewed literature.
- Major challenges and identified research gaps.

The abstract must be written as one coherent paragraph.

INDEX TERMS

Provide 4–6 relevant keywords separated by commas.

I. INTRODUCTION

Include:

- Introduction and background of the research topic.
- Importance and relevance of the research area.
- Existing problems motivating the research.
- Purpose and objectives of the review.
- Scope of the paper.
- Brief explanation of the organization of the remaining sections.

Include appropriate IEEE citations.

II. BACKGROUND AND THEORETICAL OVERVIEW

Explain the fundamental concepts required to understand the topic.

Include relevant subsections such as:

A. Definition and Fundamental Concepts
B. Architecture and Components
C. Existing Technologies and Techniques
D. Important Algorithms and Approaches

Only include subsections relevant to the research topic
and supported by the gathered information.

III. LITERATURE REVIEW

This is the main section of the review paper.

Discuss the available research papers in detail.

For each relevant research study, explain:

- Authors and publication year, if available.
- Research objectives.
- Proposed methodology or approach.
- Technologies, algorithms, or techniques used.
- Major contributions and findings.
- Limitations identified in the study.

Organize the literature into meaningful subsections based
on research themes, techniques, or approaches.

Compare related studies where appropriate.

Include IEEE citations throughout this section.

IV. COMPARATIVE ANALYSIS

Provide a critical comparison of the reviewed research studies.

Discuss:

- Similarities between existing approaches.
- Differences in methodologies and architectures.
- Strengths and weaknesses.
- Technologies and algorithms used.
- Performance or evaluation findings, if available.
- Limitations of existing approaches.

Include a comparative table using the following columns
where sufficient information is available:

| Reference | Proposed Approach | Methodology | Main Contribution | Limitations |

Explain the table in the paragraphs following it.

Do not invent performance values or comparisons.

V. CHALLENGES AND LIMITATIONS

Discuss the major challenges identified in the reviewed literature.

Consider relevant aspects such as:

- Technical challenges.
- Computational requirements.
- Accuracy and reliability.
- Scalability.
- Security and privacy.
- Integration and implementation difficulties.
- Limitations of existing methodologies.

Only discuss challenges relevant to the research topic.

VI. RESEARCH GAPS

Identify research gaps based on the reviewed studies.

Explain:

- What existing studies have not adequately addressed.
- Limitations that remain unresolved.
- Missing capabilities in existing approaches.
- Areas requiring further investigation.

Support research gaps with relevant citations.

Do not present assumptions as established research gaps.

VII. FUTURE RESEARCH DIRECTIONS

Discuss potential research opportunities and improvements.

Explain how future studies could address the identified
limitations and research gaps.

Include realistic and technically relevant directions.

VIII. CONCLUSION

Provide a comprehensive conclusion covering:

- Purpose and scope of the review.
- Main findings from the literature.
- Important research contributions.
- Comparative observations.
- Major challenges and limitations.
- Identified research gaps.
- Overall conclusion about the research area.

Do not introduce new findings or unsupported claims.

REFERENCES

Provide a complete reference list of all sources cited
in the paper.

Follow IEEE numbered referencing style.

Use the following formats where the required information
is available.

Journal Article:
[1] A. A. Author and B. B. Author, "Title of the paper,"
Journal Name, vol. X, no. X, pp. XX-XX, Year,
doi: DOI.

Conference Paper:
[2] A. A. Author, "Title of the paper," in Proc. Conference Name,
Year, pp. XX-XX, doi: DOI.

Website:
[3] Organization or Author, "Title of the webpage."
[Online]. Available: URL. [Accessed: Date].

Book:
[4] A. A. Author, Book Title, Edition.
City, Country: Publisher, Year.

REFERENCE RULES:

- Include only sources present in the gathered research.
- Use the actual bibliographic information available.
- Include original URLs whenever available.
- Include DOI information whenever available.
- Do not invent missing publication details.
- Ensure all references are numbered according to
  their first appearance in the paper.
- Ensure every in-text citation matches its reference entry.
- Do not include uncited sources in the reference list.

FINAL OUTPUT REQUIREMENTS:

1. Generate the complete review paper in the specified order.
2. Use Roman numerals for main section headings.
3. Use lettered headings for subsections.
4. Use IEEE-style numbered citations throughout.
5. Use academic paragraphs rather than excessive bullet points.
6. Include comparative tables where appropriate.
7. Keep the writing factual, detailed, coherent, and professional.
8. Do not include explanations about how you generated the report.
9. Do not add conversational introductions or closing remarks.
10. Return only the completed IEEE-style review paper.
"""),
])

writer_chain = writer_prompt | llm | StrOutputParser()





#critic_chain 


critic_prompt = ChatPromptTemplate.from_messages([

    ("system", """
You are a strict academic research paper evaluator
specializing in IEEE-style review papers.

Your task is to critically evaluate a research report
using the original research material provided.

Evaluate the report objectively. Do not give unnecessarily
high scores. Identify weaknesses honestly.

IMPORTANT EVALUATION RULES:

1. Score every metric independently from 0 to 10.
2. Use integer or decimal scores.
3. A score of 10 means excellent.
4. A score of 0 means completely inadequate.
5. Evaluate factual accuracy by comparing the report
   against the supplied research material.
6. Do not assume that a claim is correct simply because
   it sounds academically convincing.
7. Check whether citations correspond to the available sources.
8. Do not claim to have externally verified sources or DOIs.
9. Do not penalize missing information that was not available
   in the supplied research material.
10. Provide a specific explanation for every metric score.
11. Identify important missing sections and unsupported claims.
12. Keep the evaluation objective, detailed, and constructive.

"""),

    ("human", """
Evaluate the following IEEE-style research review paper.

TOPIC:
{topic}

ORIGINAL RESEARCH MATERIAL:
{research}

GENERATED RESEARCH REPORT:
{report}

Evaluate the report using the following 10 metrics.

1. RELEVANCE (0-10)
Evaluate whether the report remains focused on the research
topic and addresses its objectives.

2. RESEARCH COVERAGE (0-10)
Evaluate how comprehensively the report covers the important
aspects of the research topic.

3. FACTUAL ACCURACY (0-10)
Check whether the information, findings, and claims in the
report are supported by the original research material.
Identify unsupported or potentially fabricated statements.

4. CITATION ACCURACY (0-10)
Evaluate IEEE in-text citation usage, consistency of reference
numbers, and correspondence between citations and the
provided source information.

5. LITERATURE REVIEW QUALITY (0-10)
Evaluate how effectively the report explains existing studies,
their objectives, methodologies, contributions, and limitations.

6. COMPARATIVE ANALYSIS (0-10)
Evaluate whether different research approaches are meaningfully
compared in terms of methodology, technologies, strengths,
weaknesses, and findings.

7. CRITICAL ANALYSIS (0-10)
Evaluate whether the report provides meaningful analysis
rather than simply summarizing research papers.

8. RESEARCH GAP IDENTIFICATION (0-10)
Evaluate whether identified research gaps are logically derived
from the reviewed studies and their limitations.

9. IEEE FORMATTING (0-10)
Evaluate the report structure, section organization, headings,
academic presentation, citation style, and reference formatting.

10. CLARITY AND COHERENCE (0-10)
Evaluate academic language, readability, logical flow,
paragraph organization, and consistency.

OUTPUT FORMAT:

METRIC SCORES:

1. Relevance: X/10
Explanation: ...

2. Research Coverage: X/10
Explanation: ...

3. Factual Accuracy: X/10
Explanation: ...

4. Citation Accuracy: X/10
Explanation: ...

5. Literature Review Quality: X/10
Explanation: ...

6. Comparative Analysis: X/10
Explanation: ...

7. Critical Analysis: X/10
Explanation: ...

8. Research Gap Identification: X/10
Explanation: ...

9. IEEE Formatting: X/10
Explanation: ...

10. Clarity and Coherence: X/10
Explanation: ...

OVERALL SCORE: X/10

Calculate the overall score as the arithmetic mean
of all 10 metric scores.

STRENGTHS:
- Identify the strongest aspects of the report.

AREAS TO IMPROVE:
- Identify specific weaknesses and missing information.

RECOMMENDED IMPROVEMENTS:
- Suggest actionable improvements for the Writer Agent.

FINAL VERDICT:
Provide a brief overall assessment of the report.

Return the evaluation in the exact structure above.
""")
])

critic_chain = critic_prompt | llm | StrOutputParser()

