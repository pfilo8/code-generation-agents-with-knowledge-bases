from smolagents import fix_final_answer_code, parse_code_blobs


def extract_code(response: str) -> str:
    try:
        code_action = fix_final_answer_code(parse_code_blobs(response))
    except Exception:
        code_action = ""
    return code_action
