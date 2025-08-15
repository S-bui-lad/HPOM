import models.llm_parser as llm_parser


def parse_query(query: str, use_llm: bool = False, **kwargs):
    """Return parsed query dict: subject, relation, object, attributes, negatives, spatial_hints
    Falls back to heuristic parser bundled in models/llm_parser.
    """
    if use_llm:
        # If you have integrated an LLM client, call it here. This is a placeholder
        # Example: return llm_client.parse(query)
        try:
            return llm_parser.llm_parse(query)
        except Exception:
            return llm_parser.heuristic_parse(query)
    else:
        return llm_parser.heuristic_parse(query)