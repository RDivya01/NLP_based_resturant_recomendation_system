from rapidfuzz import process

def fuzzy_match(
    query: str,
    choices: list[str],
    threshold: int = 80,
) -> str:

    if not query:
        return query

    match = process.extractOne(
        query,
        choices,
    )

    if match and match[1] >= threshold:
        return match[0]

    return query

