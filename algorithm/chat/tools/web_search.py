from __future__ import annotations

from functools import wraps
from socket import gaierror
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urljoin

import lazyllm
import requests
from bs4 import BeautifulSoup
from httpx import ConnectError, HTTPError, HTTPStatusError, NetworkError, TimeoutException
from lazyllm import fc_register
from lazyllm.tools.tools.search import ArxivSearch, BingSearch, BochaSearch, GoogleSearch, WikipediaSearch


_MAX_TEXT_LEN = 2000
_MAX_FETCH_TEXT_LEN = 4000
_DEFAULT_WEB_SOURCES = ['bocha', 'google', 'bing', 'wikipedia']
_SUPPORTED_WEB_SOURCES = {'google', 'bing', 'bocha', 'wikipedia'}
_SCIVERSE_DEFAULT_FIELDS = [
    'title',
    'doi',
    'doc_id',
    'abstract',
    'authors',
    'publication_published_year',
    'publication_venue_name',
]
_DEFAULT_WIKIPEDIA_URLS = {
    'zh': 'https://zh.wikipedia.org',
    'en': 'https://en.wikipedia.org',
}


def _tool_failure(tool_name: str, exc: Exception) -> Dict[str, Any]:
    return {
        'success': False,
        'reason': f'{tool_name} failed: {exc}',
        'error': str(exc),
        'error_type': type(exc).__name__,
    }


def _handle_tool_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return _tool_failure(func.__name__, exc)

    return wrapper


def _agentic_config() -> Dict[str, Any]:
    config = lazyllm.globals.get('agentic_config') or {}
    return config if isinstance(config, dict) else {}


def _config_str(config: Dict[str, Any], key: str, default: str = '') -> str:
    value = config.get(key)
    return str(value if value is not None else default).strip()


def _config_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_lang(lang: str) -> str:
    normalized = str(lang or 'zh').strip().lower()
    return normalized if normalized in ('zh', 'en') else 'zh'


def _normalize_auto_sources(value: Any) -> List[str]:
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.split(',')]
    elif isinstance(value, list):
        items = [str(part).strip().lower() for part in value]
    else:
        items = list(_DEFAULT_WEB_SOURCES)
    normalized = [item for item in items if item in _SUPPORTED_WEB_SOURCES]
    return normalized or list(_DEFAULT_WEB_SOURCES)


def _truncate_text(text: Any, max_len: int = _MAX_TEXT_LEN) -> str:
    if text is None:
        return ''
    raw = text if isinstance(text, str) else str(text)
    return raw if len(raw) <= max_len else f'{raw[:max_len]}...'


def _absolute_url(url: str) -> str:
    normalized = str(url or '').strip()
    if not normalized:
        return ''
    if normalized.startswith(('http://', 'https://')):
        return normalized
    return f'https://{normalized}'


def _serialize_item(item: Dict[str, Any], content: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        'title': item.get('title', ''),
        'url': item.get('url', ''),
        'snippet': item.get('snippet', ''),
        'source': item.get('source', ''),
    }
    extra = item.get('extra')
    if isinstance(extra, dict) and extra:
        payload['extra'] = extra
    if content:
        payload['content'] = _truncate_text(content)
    return payload


def _error_details(exc: Exception) -> Dict[str, Any]:
    return {
        'error': str(exc),
        'error_type': type(exc).__name__,
    }


def _search_failure(query: str, source: str, details: Dict[str, Any], *, lang: Optional[str] = None,
                    tried_sources: Optional[List[str]] = None) -> Dict[str, Any]:
    payload = {
        'success': False,
        'status': 'search_error',
        'query': query,
        'resolved_source': source,
        'total': 0,
        'items': [],
        **details,
    }
    if lang is not None:
        payload['lang'] = lang
    if tried_sources is not None:
        payload['tried_sources'] = tried_sources
    return payload


def _classify_search_exception(exc: Exception) -> Dict[str, Any]:
    message = str(exc)
    if isinstance(exc, requests.exceptions.ConnectionError):
        return {
            'status': 'network_unreachable',
            'reason': f'search provider is unreachable: {message}',
            **_error_details(exc),
        }
    if isinstance(exc, requests.exceptions.Timeout):
        return {
            'status': 'request_timeout',
            'reason': f'search request timed out: {message}',
            **_error_details(exc),
        }
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        status_code = response.status_code if response is not None else None
        return {
            'status': 'http_error',
            'reason': f'search provider returned HTTP error{f" {status_code}" if status_code else ""}: {message}',
            'http_status': status_code,
            **_error_details(exc),
        }
    if isinstance(exc, ConnectError):
        return {
            'status': 'network_unreachable',
            'reason': f'search provider is unreachable: {message}',
            **_error_details(exc),
        }
    if isinstance(exc, (TimeoutException, gaierror)):
        return {
            'status': 'request_timeout',
            'reason': f'search request timed out or name resolution failed: {message}',
            **_error_details(exc),
        }
    if isinstance(exc, HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else None
        return {
            'status': 'http_error',
            'reason': f'search provider returned HTTP error{f" {status_code}" if status_code else ""}: {message}',
            'http_status': status_code,
            **_error_details(exc),
        }
    if isinstance(exc, (HTTPError, NetworkError)):
        return {
            'status': 'request_failed',
            'reason': f'search request failed: {message}',
            **_error_details(exc),
        }
    return {
        'status': 'search_error',
        'reason': f'search failed: {message}',
        **_error_details(exc),
    }


def _first_present(payload: Dict[str, Any], keys: Tuple[str, ...], default: Any = '') -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ''):
            return value
    return default


def _stringify_authors(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        names: List[str] = []
        for item in value:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(_first_present(item, ('name', 'display_name', 'full_name'), '')).strip()
            else:
                name = ''
            if name:
                names.append(name)
        return ', '.join(names)
    return ''


def _extract_sciverse_hits(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    containers = [payload]
    data = payload.get('data')
    if isinstance(data, dict):
        containers.append(data)

    for container in containers:
        for key in ('hits', 'items', 'results', 'papers', 'records'):
            hits = container.get(key)
            if isinstance(hits, list):
                total = container.get('total')
                if not isinstance(total, int):
                    total = container.get('total_count')
                return [hit for hit in hits if isinstance(hit, dict)], total if isinstance(total, int) else None
    return [], None


def _normalize_sciverse_item(
    item: Dict[str, Any],
    include_content: bool,
    *,
    search_type: str,
) -> Dict[str, Any]:
    title = str(_first_present(item, ('title', 'paper_title', 'name'), '')).strip()
    doi = str(_first_present(item, ('doi', 'publication_doi'), '')).strip()
    doc_id = str(_first_present(item, ('doc_id', 'document_id', 'id'), '')).strip()
    abstract = str(_first_present(item, ('abstract', 'summary', 'description'), '')).strip()
    chunk = str(_first_present(item, ('chunk', 'text', 'content'), '')).strip()
    venue = str(_first_present(item, ('publication_venue_name', 'venue', 'journal'), '')).strip()
    year = _first_present(item, ('publication_published_year', 'year', 'published_year'), '')
    score = _first_present(item, ('score', 'relevance_score'), None)
    url = str(_first_present(item, ('url', 'paper_url', 'source_url'), '')).strip()
    if not url and doi and search_type != 'agentic':
        url = f'https://doi.org/{doi}'

    snippet_parts = [part for part in (abstract, chunk) if part]
    snippet = _truncate_text('\n'.join(snippet_parts), 800)

    extra: Dict[str, Any] = {}
    for key, value in (
        ('doc_id', doc_id),
        ('doi', doi),
        ('year', year),
        ('venue', venue),
        ('authors', _stringify_authors(_first_present(item, ('authors', 'author_names'), []))),
        ('score', score),
        ('chunk_id', _first_present(item, ('chunk_id',), '')),
        ('page_no', _first_present(item, ('page_no', 'page'), '')),
        ('offset', _first_present(item, ('offset',), '')),
    ):
        if value not in (None, ''):
            extra[key] = value

    normalized = {
        'title': title,
        'snippet': snippet,
        'source': 'sciverse',
    }
    if url:
        normalized['url'] = url
    if extra:
        normalized['extra'] = extra
    if include_content and (chunk or abstract):
        normalized['content'] = _truncate_text(chunk or abstract)
    return normalized


def _sciverse_headers(api_key: str) -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'User-Agent': 'LazyMind Sciverse Search',
    }


def _sciverse_endpoint(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip('/') + '/', path.lstrip('/'))


def _run_sciverse_search(
    *,
    query: str,
    max_results: int,
    include_content: bool,
    search_type: str,
    year_from: Optional[int],
    year_to: Optional[int],
    fields: Optional[List[str]],
) -> Dict[str, Any]:
    config = _agentic_config()
    api_key = _config_str(config, 'sciverse_search_api_key')
    if not api_key:
        raise ValueError('sciverse search is not configured: missing LAZYMIND_SCIVERSE_SEARCH_API_KEY')

    base_url = _config_str(config, 'sciverse_search_base_url', 'https://api.sciverse.space')
    timeout = _config_int(config, 'sciverse_search_timeout', 15)
    limit = max(1, min(int(max_results), 10))
    normalized_type = str(search_type or 'meta').strip().lower()

    if normalized_type == 'agentic':
        endpoint = _sciverse_endpoint(base_url, '/agentic-search')
        payload: Dict[str, Any] = {'query': query, 'top_k': limit}
    elif normalized_type == 'meta':
        endpoint = _sciverse_endpoint(base_url, '/meta-search')
        filters = []
        if year_from is not None:
            filters.append({
                'field': 'publication_published_year',
                'operator': 'FILTER_OP_GTE',
                'value': int(year_from),
            })
        if year_to is not None:
            filters.append({
                'field': 'publication_published_year',
                'operator': 'FILTER_OP_LTE',
                'value': int(year_to),
            })
        payload = {
            'query': query,
            'fields': fields or list(_SCIVERSE_DEFAULT_FIELDS),
            'page': 1,
            'page_size': limit,
        }
        if filters:
            payload['filters'] = filters
    else:
        raise ValueError("search_type must be one of 'meta' or 'agentic'")

    response = requests.post(
        endpoint,
        headers=_sciverse_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError('Sciverse response must be a JSON object')

    hits, reported_total = _extract_sciverse_hits(data)
    serialized_items = [
        _normalize_sciverse_item(
            item,
            include_content=include_content,
            search_type=normalized_type,
        )
        for item in hits[:limit]
    ]
    literature_passages = [
        {
            'title': item.get('title', ''),
            'passage': item.get('content') or item.get('snippet', ''),
            'doc_id': (item.get('extra') or {}).get('doc_id', ''),
            'page_no': (item.get('extra') or {}).get('page_no', ''),
            'score': (item.get('extra') or {}).get('score', ''),
        }
        for item in serialized_items
        if item.get('content') or item.get('snippet')
    ]
    return {
        'success': True,
        'status': 'answer_ready' if literature_passages else ('ok' if serialized_items else 'no_results'),
        'query': query,
        'source': 'sciverse',
        'search_type': normalized_type,
        'total': len(serialized_items),
        'reported_total': reported_total,
        'answer_instruction': (
            'Use literature_passages to answer the user directly. Do not call url_fetch, '
            'web_search, or other web retrieval tools after this successful Sciverse result '
            'unless the user explicitly asks to inspect an original webpage.'
        ) if literature_passages else '',
        'literature_passages': literature_passages,
        'items': serialized_items,
    }


def _build_wikipedia_search(config: Dict[str, Any], lang: str) -> WikipediaSearch:
    base_url = _config_str(
        config,
        'web_search_wikipedia_base_url',
        _DEFAULT_WIKIPEDIA_URLS.get(_normalize_lang(lang), _DEFAULT_WIKIPEDIA_URLS['zh']),
    )
    timeout = _config_int(config, 'web_search_timeout', 10)
    return WikipediaSearch(base_url=base_url, timeout=timeout, source_name='wikipedia')


def _provider_available(config: Dict[str, Any], source: str) -> bool:
    if source == 'wikipedia':
        return True
    if source == 'google':
        return bool(_config_str(config, 'web_search_google_api_key')) and bool(
            _config_str(config, 'web_search_google_search_engine_id')
        )
    if source == 'bing':
        return bool(_config_str(config, 'web_search_bing_subscription_key'))
    if source == 'bocha':
        return bool(_config_str(config, 'web_search_bocha_api_key'))
    return False


def _build_provider(config: Dict[str, Any], source: str, lang: str):
    timeout = _config_int(config, 'web_search_timeout', 10)
    if source == 'wikipedia':
        return _build_wikipedia_search(config, lang)
    if source == 'google':
        api_key = _config_str(config, 'web_search_google_api_key')
        search_engine_id = _config_str(config, 'web_search_google_search_engine_id')
        if not api_key or not search_engine_id:
            raise ValueError('google search is not configured')
        return GoogleSearch(
            custom_search_api_key=api_key,
            search_engine_id=search_engine_id,
            timeout=timeout,
            source_name='google',
        )
    if source == 'bing':
        subscription_key = _config_str(config, 'web_search_bing_subscription_key')
        if not subscription_key:
            raise ValueError('bing search is not configured')
        endpoint = _config_str(config, 'web_search_bing_endpoint')
        return BingSearch(
            subscription_key=subscription_key,
            endpoint=endpoint or None,
            timeout=timeout,
            source_name='bing',
        )
    if source == 'bocha':
        api_key = _config_str(config, 'web_search_bocha_api_key')
        if not api_key:
            raise ValueError('bocha search is not configured')
        base_url = _config_str(config, 'web_search_bocha_base_url', 'https://api.bochaai.com')
        return BochaSearch(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            source_name='bocha',
        )
    raise ValueError(f'unsupported web_search source: {source}')


def _search_provider(provider: Any, source: str, query: str, topk: int) -> List[Dict[str, Any]]:
    if source == 'wikipedia':
        return provider(query, limit=topk, raise_on_error=True)[:topk]
    if source == 'google':
        return provider(query, date_restrict='', raise_on_error=True)[:topk]
    if source == 'bing':
        return provider(query, count=topk, raise_on_error=True)[:topk]
    if source == 'bocha':
        return provider(query, count=topk, summary=False, raise_on_error=True)[:topk]
    raise ValueError(f'unsupported web_search source: {source}')


def _candidate_sources(config: Dict[str, Any], requested_source: str) -> List[str]:
    if requested_source != 'auto':
        return [requested_source]

    candidates = _normalize_auto_sources(config.get('web_search_auto_sources'))
    if 'wikipedia' not in candidates:
        candidates.append('wikipedia')
    return candidates


def _run_candidate_searches(
    config: Dict[str, Any],
    source: str,
    query: str,
    topk: int,
    lang: str,
) -> Tuple[Optional[str], List[str], List[Dict[str, Any]], Optional[Any], Optional[Dict[str, Any]]]:
    requested = str(source or 'auto').strip().lower()
    tried_sources: List[str] = []
    last_error: Optional[Dict[str, Any]] = None
    last_error_source: Optional[str] = None
    last_non_error_source: Optional[str] = None

    for candidate in _candidate_sources(config, requested):
        tried_sources.append(candidate)
        if requested == 'auto' and not _provider_available(config, candidate):
            continue

        provider = _build_provider(config, candidate, lang)
        try:
            items = _search_provider(provider, candidate, query, topk)
        except Exception as exc:
            last_error = _classify_search_exception(exc)
            last_error_source = candidate
            if requested != 'auto':
                return candidate, tried_sources, [], None, last_error
            continue

        last_non_error_source = candidate
        if items:
            return candidate, tried_sources, items[:topk], provider, None
        if requested != 'auto':
            return candidate, tried_sources, [], provider, None

    resolved_source = last_non_error_source or last_error_source
    return resolved_source, tried_sources, [], None, last_error


def _content_for_item(provider: Any, item: Dict[str, Any], include_content: bool) -> Optional[str]:
    if not include_content:
        return None
    return provider.get_content(item)


def _fetch_timeout(config: Dict[str, Any]) -> int:
    return _config_int(config, 'url_fetch_timeout', _config_int(config, 'web_search_timeout', 10))


def _fetch_text_limit(config: Dict[str, Any]) -> int:
    return max(200, _config_int(config, 'url_fetch_max_length', _MAX_FETCH_TEXT_LEN))


def _extract_page_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    content_root = soup.find('main') or soup.find('article') or soup.body or soup
    lines: List[str] = []
    for node in content_root.find_all(['h1', 'h2', 'h3', 'p', 'li']):
        text = node.get_text(' ', strip=True)
        if text:
            lines.append(text)

    if not lines:
        text = content_root.get_text('\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

    deduped_lines: List[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped_lines.append(line)
    return '\n'.join(deduped_lines)


def _extract_page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    og_title = soup.find('meta', attrs={'property': 'og:title'})
    if og_title and og_title.get('content'):
        return str(og_title['content']).strip()
    return ''


def _extract_page_description(soup: BeautifulSoup) -> str:
    candidates = [
        {'name': 'description'},
        {'property': 'og:description'},
    ]
    for attrs in candidates:
        tag = soup.find('meta', attrs=attrs)
        if tag and tag.get('content'):
            return str(tag['content']).strip()
    return ''


@fc_register('tool', execute_in_sandbox=False)
@_handle_tool_errors
def web_search(
    query: str,
    source: Literal['auto', 'wikipedia', 'google', 'bing', 'bocha'] = 'auto',
    topk: int = 5,
    lang: Literal['zh', 'en'] = 'zh',
    include_content: bool = False,
) -> Dict[str, Any]:
    """Search public web information as a supplement when knowledge-base
    retrieval is insufficient.

    Prefer `kb_search` first. Use this tool only when the knowledge base
    has no relevant results, the returned evidence is clearly insufficient,
    or the user is asking for public information outside the knowledge base.

    This tool supports multiple providers through a single interface:
    `source='auto'|'wikipedia'|'google'|'bing'|'bocha'`.
    In `auto` mode, providers are tried in configured order. Unconfigured
    providers are skipped, runtime failures fall through to the next
    candidate, and Wikipedia is always appended as the final fallback.

    Args:
        query: Natural-language search query.
        source: Search provider selector. Use `auto` unless the user
            explicitly needs a specific provider.
        topk: Maximum number of result items to return.
        lang: Preferred language for Wikipedia fallback. Currently supports
            `zh` and `en`.
        include_content: Whether to fetch and include page content for each
            result item. Keep this `False` unless extra detail is necessary.

    Returns:
        A compact dict containing the resolved provider, query, and items.
    """
    normalized_query = str(query or '').strip()
    if not normalized_query:
        raise ValueError('query is required')

    config = _agentic_config()
    resolved_lang = _normalize_lang(lang)
    limit = max(1, min(int(topk), 10))
    resolved_source, tried_sources, items, provider, error = _run_candidate_searches(
        config,
        source,
        normalized_query,
        limit,
        resolved_lang,
    )
    if error is not None:
        return _search_failure(
            normalized_query,
            resolved_source or str(source),
            error,
            lang=resolved_lang,
            tried_sources=tried_sources,
        )

    serialized_items = []
    for item in items:
        content = _content_for_item(provider, item, include_content) if provider is not None else None
        serialized_items.append(_serialize_item(item, content=content))

    return {
        'success': True,
        'status': 'ok' if serialized_items else 'no_results',
        'query': normalized_query,
        'requested_source': source,
        'resolved_source': resolved_source or str(source),
        'tried_sources': tried_sources,
        'lang': resolved_lang,
        'total': len(serialized_items),
        'items': serialized_items,
    }


@fc_register('tool', execute_in_sandbox=False)
@_handle_tool_errors
def arxiv_search(
    query: str,
    max_results: int = 5,
    include_content: bool = False,
    sort_by: Literal['relevance', 'lastUpdatedDate', 'submittedDate'] = 'relevance',
) -> Dict[str, Any]:
    """Search arXiv papers for academic questions such as paper titles,
    authors, abstracts, or arXiv identifiers.

    Prefer this tool over `web_search` when the user is asking about papers,
    research topics, or arXiv records.

    Args:
        query: Paper title, topic, author keywords, or arXiv id related text.
        max_results: Maximum number of result items to return.
        include_content: Whether to include the paper abstract text in the
            returned items.
        sort_by: arXiv sort field.

    Returns:
        A compact dict with arXiv search results.
    """
    normalized_query = str(query or '').strip()
    if not normalized_query:
        raise ValueError('query is required')

    config = _agentic_config()
    timeout = _config_int(config, 'arxiv_search_timeout', 15)
    limit = max(1, min(int(max_results), 10))
    provider = ArxivSearch(timeout=timeout, source_name='arxiv')
    try:
        items = provider(
            normalized_query,
            max_results=limit,
            sort_by=sort_by,
            raise_on_error=True,
        )[:limit]
    except Exception as exc:
        return _search_failure(normalized_query, 'arxiv', _classify_search_exception(exc))

    serialized_items = []
    for item in items:
        content = _content_for_item(provider, item, include_content)
        serialized_items.append(_serialize_item(item, content=content))

    return {
        'success': True,
        'status': 'ok' if serialized_items else 'no_results',
        'query': normalized_query,
        'source': 'arxiv',
        'sort_by': sort_by,
        'total': len(serialized_items),
        'items': serialized_items,
    }


@fc_register('tool', execute_in_sandbox=False)
@_handle_tool_errors
def sciverse_search(
    query: str,
    max_results: int = 5,
    include_content: bool = True,
    search_type: Literal['meta', 'agentic'] = 'agentic',
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> Dict[str, Any]:
    """Search Sciverse scientific literature records.

    Prefer this tool over `web_search` for paper titles, research topics,
    author-related questions, abstracts, DOI/metadata lookup, or scientific
    literature retrieval after knowledge-base evidence is unavailable or
    insufficient. This is a terminal retrieval tool for scientific question
    answering: when it returns `status='answer_ready'`, immediately answer
    from `literature_passages` and do not call `url_fetch`, `web_search`, or
    another web retrieval tool. Use the default `search_type='agentic'` for
    scientific question answering because it returns citable literature
    snippets. Use `search_type='meta'` only for bibliographic lists or
    structured metadata lookup.

    Args:
        query: Paper title, topic, author keywords, DOI, or scientific query.
        max_results: Maximum number of result items to return.
        include_content: Whether to include abstract/chunk text in `content`.
        search_type: `meta` returns bibliographic metadata; `agentic`
            returns citable text chunks when the query needs passage evidence.
        year_from: Optional inclusive lower bound for publication year.
        year_to: Optional inclusive upper bound for publication year.

    Returns:
        A compact dict with Sciverse search results.
    """
    normalized_query = str(query or '').strip()
    if not normalized_query:
        raise ValueError('query is required')

    try:
        return _run_sciverse_search(
            query=normalized_query,
            max_results=max_results,
            include_content=include_content,
            search_type=search_type,
            year_from=year_from,
            year_to=year_to,
            fields=None,
        )
    except Exception as exc:
        return _search_failure(
            normalized_query,
            'sciverse',
            _classify_search_exception(exc),
        )


@fc_register('tool', execute_in_sandbox=False)
@_handle_tool_errors
def url_fetch(
    url: str,
) -> Dict[str, Any]:
    """Fetch and summarize the readable content of a public web page.

    Use this when the user provides a concrete URL or when search results
    already identified a page that needs direct inspection.

    Args:
        url: Absolute URL, or a domain/path that can be normalized to HTTPS.

    Returns:
        A compact dict containing page metadata and extracted text content.
    """
    normalized_url = _absolute_url(url)
    if not normalized_url:
        raise ValueError('url is required')

    config = _agentic_config()
    timeout = _fetch_timeout(config)
    text_limit = _fetch_text_limit(config)
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        )
    }

    with requests.sessions.Session() as session:
        response = session.get(
            normalized_url,
            timeout=timeout,
            headers=headers,
            allow_redirects=True,
        )
        response.raise_for_status()

    content_type = str(response.headers.get('Content-Type') or '').lower()
    if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
        raw_text = response.text.strip()
        return {
            'success': True,
            'status': 'ok',
            'url': normalized_url,
            'final_url': response.url,
            'status_code': response.status_code,
            'content_type': content_type,
            'title': '',
            'description': '',
            'content': _truncate_text(raw_text, text_limit),
        }

    soup = BeautifulSoup(response.text, 'html.parser')
    return {
        'success': True,
        'status': 'ok',
        'url': normalized_url,
        'final_url': response.url,
        'status_code': response.status_code,
        'content_type': content_type,
        'title': _extract_page_title(soup),
        'description': _truncate_text(_extract_page_description(soup), 500),
        'content': _truncate_text(_extract_page_text(response.text), text_limit),
    }
