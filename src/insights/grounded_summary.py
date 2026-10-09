def experiment_prompt(results: dict) -> str:
    """LLM-ready prompt: interpretation only; statistics must already be computed."""
    return f'''You are an experiment insights analyst. Interpret only the supplied calculated results. Do not recompute or invent statistics.\nRESULTS: {results}\nExplain: outcome, uncertainty, strongest segment effects, rollout recommendation, and caveats in <=150 words.'''
