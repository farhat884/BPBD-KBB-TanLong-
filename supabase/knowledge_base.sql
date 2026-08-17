-- BPBD KBB - Supabase Knowledge Base
create extension if not exists vector;

create table if not exists ai_documents (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    filename text not null,
    storage_path text not null unique,
    file_type text,
    file_size bigint,
    uploaded_by text,
    uploaded_at timestamptz default now(),
    status text default 'processing',
    chunk_count integer default 0,
    error_message text
);

create table if not exists ai_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references ai_documents(id) on delete cascade,
    chunk_index integer not null,
    content text not null,
    source text,
    created_at timestamptz default now(),
    unique(document_id, chunk_index)
);

create index if not exists idx_ai_chunks_document_id
on ai_chunks(document_id);

create index if not exists idx_ai_documents_status
on ai_documents(status);
