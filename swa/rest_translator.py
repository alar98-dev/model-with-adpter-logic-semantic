"""Translator that maps incoming tool/schema requests into adapter inputs or triplets.

This is a lightweight module (not a web server). Use it to convert structured requests
into the internal representation used by the SWA training/engine.
"""
from typing import Dict, Tuple


def schema_to_triplet(schema: Dict) -> Tuple[str, str, str]:
    """Convert a simple schema into (anchor, positive, negative).

    Expected schema example:
      {"claim": "A causa x leva a y", "preserve": "rewritten logical", "break": "contradictory text"}
    """
    anchor = schema.get('claim') or schema.get('anchor') or ''
    positive = schema.get('preserve') or schema.get('positive') or anchor
    negative = schema.get('break') or schema.get('negative') or ''
    return anchor, positive, negative


def ensure_schema_compat(schema: Dict) -> Dict:
    """Normalize incoming schema to a small canonical form.

    This accepts variants used by external tooling and maps them to
    the expected keys: 'claim', 'preserve', 'break'. It is safe to call
    before `schema_to_triplet`.
    """
    s = dict(schema)
    # Common aliases
    if 'text' in s and 'claim' not in s:
        s['claim'] = s.pop('text')
    if 'anchor' in s and 'claim' not in s:
        s['claim'] = s.get('anchor')
    if 'positive' in s and 'preserve' not in s:
        s['preserve'] = s.get('positive')
    if 'negative' in s and 'break' not in s:
        s['break'] = s.get('negative')
    return s


def tool_schema_to_text(schema: Dict) -> str:
    """Serialize a tool/schema definition (OpenAPI-like) into a human-readable text.

    This handles objects that include keys like `name`, `description`, and
    `parameters` (with `properties`), and produces a compact textual form
    suitable for embedding.
    """
    parts = []
    name = schema.get('name') or schema.get('title')
    if name:
        parts.append(f"Tool: {name}")
    desc = schema.get('description')
    if desc:
        parts.append(f"Description: {desc}")

    # parameters may be provided in various forms
    params = schema.get('parameters') or schema.get('properties') or schema.get('params')
    if isinstance(params, dict):
        parts.append("Parameters:")
        required = schema.get('required', [])
        for k, v in params.items():
            if isinstance(v, dict):
                ptype = v.get('type') or v.get('schema', {}).get('type') if v.get('schema') else v.get('type')
                pdesc = v.get('description') or v.get('title') or ''
            else:
                ptype = str(type(v))
                pdesc = ''
            req_flag = "(required)" if k in required else ""
            parts.append(f" - {k}: {ptype} {req_flag} {pdesc}".strip())
    elif isinstance(params, list):
        parts.append("Parameters:")
        for p in params:
            if isinstance(p, dict):
                pname = p.get('name')
                pdesc = p.get('description', '')
                ptype = p.get('schema', {}).get('type') if p.get('schema') else p.get('type', '')
                req_flag = "(required)" if p.get('required') else ""
                parts.append(f" - {pname}: {ptype} {req_flag} {pdesc}".strip())

    # fallback: include raw schema keys
    if not parts:
        return str(schema)
    return "\n".join(parts)


def request_to_adapter_input(req: Dict) -> str:
    """Flatten a request with tools/schema into a single text for embedding.

    This function can be customized to serialize tool outputs, context and metadata.
    """
    parts = []
    if 'context' in req:
        parts.append(req['context'])
    if 'instruction' in req:
        parts.append(req['instruction'])
    if req.get('tools'):
        # tools is expected to be a dict of tool_name->output
        for k, v in req['tools'].items():
            parts.append(f"[{k}]: {v}")
    return '\n'.join(parts)


def normalize_and_flatten_request(req: Dict) -> str:
    """Helper that ensures schema compatibility and flattens into a text.

    This is convenient when callers send varied tool-schema shapes and
    we want a stable string for embedding by the `Vectorizer`.
    """
    req = dict(req)
    if 'schema' in req and isinstance(req['schema'], dict):
        schema = ensure_schema_compat(req['schema'])
        # If schema looks like a tool description (has parameters/name), serialize accordingly
        if 'parameters' in schema or 'properties' in schema or 'name' in schema:
            return tool_schema_to_text(schema)
        # otherwise keep triplet-style normalization
        req['schema'] = schema
    # if tools is provided as a list of tool schemas, flatten them
    if 'tools' in req and isinstance(req['tools'], list):
        tool_texts = [tool_schema_to_text(t) if isinstance(t, dict) else f"[{t}]" for t in req['tools']]
        # attach to context
        req.setdefault('context', '')
        req['context'] = (req['context'] + '\n' + '\n'.join(tool_texts)).strip()
    return request_to_adapter_input(req)
