from __future__ import annotations

import json
import shlex
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

from config import config as _cfg

MEMORY_REVIEW_SCHEMA = 'public'
MEMORY_REVIEW_TABLE = 'memory_review'
MEMORY_REVIEW_TABLE_QUALIFIED = f'{MEMORY_REVIEW_SCHEMA}.{MEMORY_REVIEW_TABLE}'
MEMORY_REVIEW_TARGET_CHECK = f'{MEMORY_REVIEW_TABLE}_target_check'

_DB_URL_ENV_HINT = (
    'LAZYMIND_CORE_DATABASE_URL, LAZYMIND_ACL_DB_DSN, or LAZYMIND_DATABASE_URL'
)
_engine_cache: Dict[str, Engine] = {}
_engine_cache_lock = threading.Lock()
_table_ensured = False
_table_ensure_lock = threading.Lock()


def _ensure_postgres_driver(url: str) -> str:
    normalized = url.strip()
    parts = urlsplit(normalized)
    scheme = (parts.scheme or '').lower()
    if scheme in {'postgresql', 'postgres'}:
        return urlunsplit((
            f'{scheme}+psycopg2',
            parts.netloc,
            parts.path,
            parts.query,
            parts.fragment,
        ))
    return normalized


def _dsn_to_sqlalchemy_url(dsn: str) -> str:
    if '://' in dsn:
        return _ensure_postgres_driver(dsn)

    parts: Dict[str, str] = {}
    for token in shlex.split(dsn):
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        parts[key.strip()] = value.strip()

    if not parts:
        raise ValueError('invalid database dsn')
    if not (parts.get('host') or '').strip():
        raise ValueError('database host is required')
    database = (parts.get('dbname') or parts.get('database') or '').strip()
    if not database:
        raise ValueError('database name is required')
    try:
        port = int(parts['port']) if parts.get('port') else 5432
    except ValueError as exc:
        raise ValueError('invalid database port') from exc

    return str(URL.create(
        'postgresql+psycopg2',
        username=parts.get('user') or None,
        password=parts.get('password') or None,
        host=parts['host'],
        port=port,
        database=database,
    ))


def _normalize_pg_url(url: Optional[str] = None, dsn: Optional[str] = None) -> str:
    if dsn and dsn.strip():
        return _dsn_to_sqlalchemy_url(dsn)
    if url and url.strip():
        return _ensure_postgres_driver(url)
    raise RuntimeError(f'postgres connection config is required: {_DB_URL_ENV_HINT}')


def _get_engine(*, url: Optional[str] = None, dsn: Optional[str] = None) -> Engine:
    engine_url = _normalize_pg_url(url=url, dsn=dsn)
    engine = _engine_cache.get(engine_url)
    if engine is not None:
        return engine
    with _engine_cache_lock:
        engine = _engine_cache.get(engine_url)
        if engine is None:
            engine = create_engine(engine_url, future=True, pool_pre_ping=True)
            _engine_cache[engine_url] = engine
    return engine


def _resolve_memory_review_conn_target() -> tuple[Optional[str], Optional[str]]:
    core_db_url = _cfg['core_database_url']
    if core_db_url and core_db_url.strip():
        return core_db_url.strip(), None

    core_db_dsn = _cfg['acl_db_dsn']
    if core_db_dsn and core_db_dsn.strip():
        return None, core_db_dsn.strip()

    db_url = _cfg['database_url']
    if db_url and db_url.strip():
        return db_url.strip(), None
    return None, None


def _get_memory_review_conn() -> Engine:
    url, dsn = _resolve_memory_review_conn_target()
    if not (url or dsn):
        raise RuntimeError(
            f'[MemoryReviewDB] {_DB_URL_ENV_HINT} is not set; cannot write memory_review.'
        )
    return _get_engine(url=url, dsn=dsn)


def ensure_memory_review_table() -> None:
    engine = _get_memory_review_conn()
    with engine.begin() as conn:
        conn.execute(text(
            f"""
            CREATE TABLE IF NOT EXISTS {MEMORY_REVIEW_TABLE_QUALIFIED} (
                id TEXT PRIMARY KEY,
                target TEXT NOT NULL CHECK (target IN ('memory', 'user_preference')),
                session_id TEXT NOT NULL,
                source_content TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                operations JSONB NOT NULL DEFAULT '[]'::jsonb,
                state TEXT NOT NULL DEFAULT 'success',
                review_status TEXT NOT NULL DEFAULT 'pending',
                time TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ))
        conn.execute(text(
            f"""
            ALTER TABLE {MEMORY_REVIEW_TABLE_QUALIFIED}
                DROP CONSTRAINT IF EXISTS {MEMORY_REVIEW_TARGET_CHECK}
            """
        ))
        conn.execute(text(
            f"""
            ALTER TABLE {MEMORY_REVIEW_TABLE_QUALIFIED}
                ADD CONSTRAINT {MEMORY_REVIEW_TARGET_CHECK}
                CHECK (target IN ('memory', 'user_preference'))
            """
        ))
        conn.execute(text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_memory_review_session_target_time
                ON {MEMORY_REVIEW_TABLE_QUALIFIED} (session_id, target, time DESC)
            """
        ))


def _ensure_table_once() -> None:
    global _table_ensured
    if _table_ensured:
        return
    with _table_ensure_lock:
        if not _table_ensured:
            ensure_memory_review_table()
            _table_ensured = True


def insert_memory_review_record(
    *,
    target: str,
    session_id: str,
    content: str,
    source_content: str = '',
    operations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if target not in {'memory', 'user_preference'}:
        raise ValueError("target must be one of 'memory' or 'user_preference'.")
    if not session_id.strip():
        raise ValueError('session_id is required.')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('content must be a non-empty string.')

    _ensure_table_once()

    record_id = str(uuid4())
    operation_payload = json.dumps(operations or [], ensure_ascii=False)
    created_at = datetime.now(timezone.utc).isoformat()

    engine = _get_memory_review_conn()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO {MEMORY_REVIEW_TABLE_QUALIFIED} (
                    id,
                    target,
                    session_id,
                    source_content,
                    content,
                    operations,
                    state,
                    review_status,
                    time
                )
                VALUES (
                    :id,
                    :target,
                    :session_id,
                    :source_content,
                    :content,
                    CAST(:operations AS JSONB),
                    'success',
                    'pending',
                    :time
                )
                """
            ),
            {
                'id': record_id,
                'target': target,
                'session_id': session_id.strip(),
                'source_content': source_content,
                'content': content,
                'operations': operation_payload,
                'time': created_at,
            },
        )

    return {
        'id': record_id,
        'target': target,
        'session_id': session_id.strip(),
        'state': 'success',
        'review_status': 'pending',
        'time': created_at,
    }
