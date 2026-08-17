# Supabase integration - BPBD KBB

## Environment variables (Vercel)
Required:
- SUPABASE_URL
- SUPABASE_PUBLISHABLE_KEY
- SUPABASE_SECRET_KEY
- SUPABASE_BUCKET=ai-guides (optional; defaults to ai-guides)
- GROQ_API_KEY
- FIREBASE_CONFIG_JSON
- FIREBASE_DB_URL
- FLASK_SECRET_KEY

Never put SUPABASE_SECRET_KEY in frontend JavaScript.

## Supabase Storage
Create a bucket named `ai-guides` and keep it PRIVATE.
Recommended allowed file types:
- application/pdf
- application/vnd.openxmlformats-officedocument.wordprocessingml.document
- text/plain
- text/markdown

Free plan: keep the bucket/file limit at or below 50 MB.

## Database
The project expects:
- ai_documents
- ai_chunks

The SQL used to create these tables is included below.

## Upload flow
1. Admin browser sends only filename/size/CSRF token to Vercel.
2. Flask creates a time-limited Supabase signed upload URL.
3. Browser uploads the document directly to Supabase Storage.
4. Browser sends only the Storage path back to Flask.
5. Flask downloads the file from private Storage, extracts text, chunks it, and stores chunks in Supabase Postgres.
6. RAG retrieval searches the stored chunks and sends relevant context to Groq.

This avoids Vercel Serverless Function request-body limits for large document uploads.

## Current retrieval
This version uses TF-IDF retrieval over chunks stored in Supabase. The pgvector extension can remain enabled; semantic embeddings can be added later without changing the document/storage flow.
