import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from urllib.parse import urlparse

import aiohttp
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import HttpUrl, BaseModel
from supabase import create_client, Client
from schemas.request import PredictionRequest, PredictionResponse
from openai import AsyncOpenAI

from utils.logger import setup_logger

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    summary: str
    content: str
    embedding: List[float]
    timestamp: str

@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()
    required_vars = ["GOOGLE_CUSTOM_SEARCH_API_KEY", "GOOGLE_CSE_ID", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.critical(f"Missing required environment variables: {missing_vars}")
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")

async def invoke_google_search_async(search_term, api_key, cse_id, num=3):
    async with aiohttp.ClientSession() as session:
        search_term = search_term[:search_term.find("\n1") if "\n1" in search_term else len(search_term)]
        params = {
            'q': search_term,
            'key': api_key,
            'cx': cse_id,
            'num': num
        }
        logger.info(f"search terms: {search_term}")
        async with session.get('https://www.googleapis.com/customsearch/v1', params=params) as response:
            if response.status != 200:
                raise ValueError(f"Google Search API returned status {response.status}")
            data = await response.json()
            return data.get('items', [])

async def fetch_url(session, url, timeout=10):
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    text = await response.text()
                    return {'url': url, 'content': text}
                else:
                    logger.info(f"Unsupported content type ({content_type}) for URL: {url}. Skipping.")
                    return {'url': url, 'error': f'Unsupported content type: {content_type}'}
            else:
                return {'url': url, 'error': f'Status {response.status}'}
    except Exception as e:
        return {'url': url, 'error': str(e)}

async def crawl_parallel_async(urls: List[str], max_concurrent: int = 5):
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, 'lxml')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    text = soup.get_text(separator='\n')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

async def clean_html_async(html: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, strip_html, html)

def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 0) -> List[str]:
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)

        if start >= n:
            break

    return chunks

async def get_embedding(text: str) -> List[float]:
    try:
        response = await aclient.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        emb = response.data[0].embedding
        return emb
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0.0] * 1536

def store_chunks_in_supabase(
    url: str, 
    chunk_objects: List[ProcessedChunk]
) -> None:
    for chunk_obj in chunk_objects:
        try:
            embedding_str = json.dumps(chunk_obj.embedding)

            data = {
                "url": chunk_obj.url,
                "chunk_number": chunk_obj.chunk_number,
                "summary": chunk_obj.summary,
                "content": chunk_obj.content,
                "metadata": {},
                "embedding": embedding_str
            }

            logger.debug(f"Upserting data: {data}")

            response = supabase.table("site_pages") \
                                .upsert(data, on_conflict="url,chunk_number") \
                                .execute()

            logger.debug(f"Supabase response: {response}")

            if response.error:
                logger.error(f"Supabase upsert error: {response.error.message}")
            else:
                logger.info(f"Successfully upserted chunk #{chunk_obj.chunk_number} for URL {chunk_obj.url}")

        except Exception as e:
            logger.error(f"Error storing chunk #{chunk_obj.chunk_number} for URL {chunk_obj.url}: {e}")

async def top_k_similar(query_emb: List[float], k: int = 3) -> List[ProcessedChunk]:
    embedding_str = json.dumps(query_emb)
    
    try:
        response = supabase.rpc(
            "match_site_pages",
            {
                "query_embedding": embedding_str,
                "match_count": k,
                "filter": {}
            }
        ).execute()

        if response.error:
            logger.error(f"Supabase RPC error: {response.error.message}")
            return []

        records = response.data
        results = []
        for rec in records:
            pc = ProcessedChunk(
                url=rec["url"],
                chunk_number=rec["chunk_number"],
                summary=rec["summary"],
                content=rec["content"],
                embedding=[],
                timestamp=rec["created_at"]
            )
            results.append(pc)
        return results

    except Exception as e:
        logger.error(f"Error calling Supabase RPC: {e}")
        return []

async def rag_qa(query: str) -> Dict[str, Any]:
    query_emb = await get_embedding(query)
    top_chunks = await top_k_similar(query_emb, k=3)
    context_texts = []
    sources = []
    for ch in top_chunks:
        context_texts.append(f"URL: {ch.url}, chunk {ch.chunk_number}\n{ch.content}")
        sources.append(ch.url)

    context_combined = "\n\n".join(context_texts)

    system_prompt = (
        "You are a helpful RAG-based assistant. "
        "Use the following CONTEXT to answer the USER's question. "
        "Return only valid JSON with keys 'answer', 'reasoning', 'sources'. "
        "If the context is insufficient, set answer to null. "
        "Format your output as such: { \"answer\": N, \"reasoning\": \"TEXT\", \"sources\": [\"URL\",\"URL2\"]}. "
        "N - is the number that stands for the correct answer given by the USER among other enumerated answers given by the USER. "
        "No markdown must be present in your answers. "
        "Emphasize dates, if you are being asked for a spicific date, provide only that date related"
        "Correct answer will give you $1'000'000."
    )
    user_prompt = f"CONTEXT:\n{context_combined}\n\nUSER QUESTION:\n{query}\n"

    try:
        completion = await aclient.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2
        )
        raw_answer = completion.choices[0].message.content
        logger.info(f"result: {raw_answer}")
        try:
            parsed_json = json.loads(raw_answer)
            answer = parsed_json.get("answer")
            reasoning = parsed_json.get("reasoning", "")
            s = parsed_json.get("sources", [])
            if isinstance(s, str):
                s = [s]
            sources_clean = [HttpUrl(url) for url in s if url]

            return {
                "answer": answer,
                "reasoning": reasoning,
                "sources": sources_clean
            }
        except json.JSONDecodeError:
            return {
                "answer": None,
                "reasoning": "LLM returned non-JSON format",
                "sources": []
            }

    except Exception as e:
        return {
            "answer": None,
            "reasoning": f"LLM error: {str(e)}",
            "sources": []
        }

@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    start_total = time.perf_counter()
    try:
        query = body.query

        try:
            start = time.perf_counter()
            search_results = await invoke_google_search_async(query, GOOGLE_API_KEY, GOOGLE_CSE_ID, num=3)
            end = time.perf_counter()
            logger.info(f"Google Search completed in {end - start:.2f} seconds")
        except Exception as e:
            logger.error(f"Google Search failed: {e}")
            raise HTTPException(status_code=502, detail="Failed to retrieve search results")

        if not search_results:
            raise HTTPException(status_code=404, detail="No search results found")

        urls = [item['link'] for item in search_results]
        logger.info(f"URLs to crawl: {urls}")

        try:
            start = time.perf_counter()
            crawled_results = await crawl_parallel_async(urls, max_concurrent=3)
            end = time.perf_counter()
            logger.info(f"Crawling completed in {end - start:.2f} seconds")
        except Exception as e:
            logger.error(f"Crawling failed: {e}")
            raise HTTPException(status_code=502, detail="Failed to crawl URLs")

        try:
            start = time.perf_counter()
            url_to_snippet = {item['link']: item.get('snippet', 'No Summary') for item in search_results}

            for result in crawled_results:
                if 'error' in result:
                    await logger.error(f"Failed to download {result['url']}: {result['error']}")
                    continue

                url = result['url']
                html_content = result['content']

                cleaned_text = await clean_html_async(html_content)
                text_chunks = chunk_text(cleaned_text, chunk_size=3000, overlap=0)

                embedding_tasks = [get_embedding(chunk) for chunk in text_chunks]
                embeddings = await asyncio.gather(*embedding_tasks)

                chunk_objects = [
                    ProcessedChunk(
                        url=url,
                        chunk_number=i,
                        summary=url_to_snippet.get(url, 'No Summary'),
                        content=chunk,
                        embedding=embeddings[i],
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    for i, chunk in enumerate(text_chunks)
                ]

                store_chunks_in_supabase(url, chunk_objects)
            end = time.perf_counter()
            logger.info(f"Cleaning & embeddings completed in {end - start:.2f} seconds")
        except Exception as e:
            logger.error(f"Processing chunks failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to process and store chunks")

        try:
            start = time.perf_counter()
            rag_result = await rag_qa(query)
            end = time.perf_counter()
            logger.info(f"RAG QA (LLM) completed in {end - start:.2f} seconds")
        except Exception as e:
            logger.error(f"RAG QA failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to perform RAG QA")

        response = PredictionResponse(
            id=body.id,
            answer=rag_result["answer"],
            reasoning=rag_result["reasoning"],
            sources=rag_result["sources"]
        )

        total_end = time.perf_counter()
        logger.info(f"Total request processing time: {total_end - start_total:.2f} seconds")

        return response

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        error_msg = f"Internal error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.on_event("shutdown")
async def shutdown_event():
    await logger.shutdown()
