from src.pipeline.pipeline import run_research_pipeline

topic=input("enter a topic: ")
run_research_pipeline(topic)
# from rich import print
# import os
# from datetime import datetime


# # ─────────────────────────────────────────────────────────ai 
# # SAVE HELPER
# # ─────────────────────────────────────────────────────────quantum computing 
# def save_to_file(data, filename=None):
#     """Create a file and write incoming data to it."""
#     desktop = os.path.join(os.path.expanduser("~"), "Desktop")
#     os.makedirs(desktop, exist_ok=True)

#     if filename is None:
#         filename = f"scraped_{datetime.now():%Y%m%d_%H%M%S}.txt"

#     path = os.path.join(desktop, filename)

#     with open(path, "w", encoding="utf-8") as f:
#         f.write(str(data))

#     print(f"\n✅ Saved to: {path}")
#     return path


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
# result = scrape_url.invoke({
#     "url": "https://arxiv.org/pdf/2411.11901"
# })

# # result = web_search.invoke("ai and quantum computing")

# print(result)

# Save the scraped data to a file on the Desktop
# save_to_file(result, filename="scraped_arxiv_2411.11901.txt")

