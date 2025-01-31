create extension if not exists vector;

create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),  -- OpenAI 1536 text-embedding-3
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,

    unique(url, chunk_number)
);

create index on site_pages using ivfflat (embedding vector_cosine_ops);

create index idx_site_pages_metadata on site_pages using gin (metadata);

create function match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb default '{}'::jsonb
) 
returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Включаем RLS (row-level security),
alter table site_pages enable row level security;

create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);
