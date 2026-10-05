import os

from fastapi import FastAPI, Query
from pydantic import BaseModel

from get_data import get_data
from videoservice import get_videos_by_creator
from bson import json_util
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma


# =========================================================
# 1. FASTAPI APP
# =========================================================

app = FastAPI(
    title="Creator AI Assistant",
    description="RAG-based AI assistant for creator video data",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

frontend_url = os.getenv("FRONTEND_URL", "https://creator-os-frontend-six.vercel.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 2. REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    creatorId: str
    question: str


class TopicRequest(BaseModel):
    creatorId: str
    category: str
    target_audience: str
    tone: str
    goal: str


# =========================================================
# 3. MODELS (ek hi baar banay, reuse hote hain)
# =========================================================

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="conversational",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=200,
    temperature=0.7,
)

chat_bot = ChatHuggingFace(llm=llm)


# =========================================================
# 4. PROMPTS
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI Assistant for a content creator.
Use ONLY the provided context to answer the user's question.
If the answer cannot be determined from the context,
clearly say that there is not enough information.

Context:
{context}
"""),
    ("human", "{question}")
])

trend_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a YouTube growth strategist. Analyze the creator's top-performing
    videos below and identify patterns: common topics, title styles, tags, or
    themes that correlate with higher views.

    Give 3-4 concise, actionable insights (not generic advice). Format as a
    short bullet list using '-' at the start of each line.
    """),
    ("human", "Top performing videos:\n{videos}")
])

suggestion_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a YouTube content strategist. Based on the creator's past video
    titles and their stated preferences, suggest ONE specific, compelling
    video topic with a suggested title.

    Keep it practical and tailored, not generic.
    """),
    ("human", """
    Past video titles:
    {past_titles}

    Creator's preferences:
    Category: {category}
    Target Audience: {target_audience}
    Tone: {tone}
    Goal: {goal}

    Suggest a new video topic.
    """)
])


# =========================================================
# 5. CONVERT VIDEOS INTO DOCUMENTS
# =========================================================

def build_documents(videos):
    documents = []
    for video in videos:
        stats_list = video.get("stats", [])
        stats = stats_list[0] if stats_list else {}

        content = f"""
Video Title: {video.get('title', 'Untitled')}
Description: {video.get('description') or 'No description provided'}
Published: {video.get('video_published', 'Unknown date')}
Views: {stats.get('views', 0)}
Likes: {stats.get('likes', 0)}
Comments: {stats.get('comments', 0)}
""".strip()

        documents.append(Document(
            page_content=content,
            metadata={"video_id": video.get("video_id"), "title": video.get("title")}
        ))
    return documents


# =========================================================
# 6. PER-CREATOR VECTORSTORE / RETRIEVER
# =========================================================

def get_retriever_for_creator(creator_id: str):
    result = get_data(creatorId=creator_id)
    data = result["data"]

    if not data:
        return None

    documents = build_documents(data)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name=f"creator_{creator_id}",
        persist_directory="./chroma_db"
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )


# =========================================================
# 7. FEATURE 1 — CHAT (Quick Questions)
# =========================================================

def ask_question(creator_id: str, question: str):
    retriever = get_retriever_for_creator(creator_id)

    if retriever is None:
        return "No video data found for this creator."

    relevant_docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    response = chat_bot.invoke(final_prompt)

    return response.content


# =========================================================
# 8. FEATURE 2 — TREND INSIGHTS
# =========================================================

def get_trend_insights(creator_id: str):
    result = get_data(creatorId=creator_id)
    data = result["data"]

    if not data:
        return "No video data available to analyze."

    sorted_videos = sorted(
        data,
        key=lambda v: int(v.get('stats', [{}])[0].get('views', 0)) if v.get('stats') else 0,
        reverse=True
    )
    top_videos = sorted_videos[:6]

    videos_summary = "\n".join([
        f"- '{v.get('title')}' | Views: {v.get('stats', [{}])[0].get('views', 0) if v.get('stats') else 0} | Tags: {', '.join(v.get('tags', []))}"
        for v in top_videos
    ])

    final_prompt = trend_prompt.invoke({"videos": videos_summary})
    response = chat_bot.invoke(final_prompt)

    return response.content


# =========================================================
# 9. FEATURE 3 — SUGGEST NEW TOPIC
# =========================================================

def suggest_new_topic(creator_id: str, answers: dict):
    result = get_data(creatorId=creator_id)
    data = result["data"]

    past_titles = "\n".join([f"- {v.get('title')}" for v in data[:10]]) if data else "No past videos available"

    final_prompt = suggestion_prompt.invoke({
        "past_titles": past_titles,
        "category": answers.get("category", "Not specified"),
        "target_audience": answers.get("target_audience", "Not specified"),
        "tone": answers.get("tone", "Not specified"),
        "goal": answers.get("goal", "Not specified"),
    })

    response = chat_bot.invoke(final_prompt)
    return response.content


# =========================================================
# 10. ROUTES
# =========================================================

@app.get("/")
def home():
    return {"status": "success", "message": "Creator AI Assistant API is running"}


@app.get("/api/data")
def get_data_route(creatorId: str = Query(...)):
    raw_data = get_videos_by_creator(creator_id=creatorId)

    if raw_data is None:
        return {"success": False, "message": "Invalid creatorId format"}

    data = json.loads(json_util.dumps(raw_data))
    return {"success": True, "data": data}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = ask_question(request.creatorId, request.question)
        print(answer)
        return {
            "success": True,
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/trends")
def trends_route(creatorId: str = Query(...)):
    try:
        insights = get_trend_insights(creator_id=creatorId)
        return {"success": True, "insights": insights}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/suggest-topic")
def suggest_topic_route(request: TopicRequest):
    try:
        answers = {
            "category": request.category,
            "target_audience": request.target_audience,
            "tone": request.tone,
            "goal": request.goal,
        }
        suggestion = suggest_new_topic(creator_id=request.creatorId, answers=answers)
        return {"success": True, "suggestion": suggestion}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/health")
def health():
    return {"status": "healthy"}


# =========================================================
# 11. LOCAL TEST
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)