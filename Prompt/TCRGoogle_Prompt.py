def build_tcr_prompt(search_result_text: str) -> list:
    """
    Constructs a prompt for immunological interpretation of TCR search results.

    Args:
        search_result_text (str): Raw text output from querying TCR databases.

    Returns:
        list: A list of messages formatted for chat-based LLMs.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional immunologist."
        },
        {
            "role": "user",
            "content": f"""Below is the result of querying a T-cell receptor (TCR) sequence in several immunological databases.

### Search Results ###
{search_result_text}
### End of Search Results ###

Please analyze the data and respond with your expert interpretation in the following format:

Antigen(s): <comma-separated list of antigen peptide sequences>  
Justification: <brief explanation referring to database match, confidence, MHC type, disease relevance, etc.>

Instructions:
- Include all plausible antigen sequences.
- Refer specifically to which databases matched and the confidence level.
- Mention any known disease associations (virus, tumor, autoimmune).
- You MUST respond ONLY in this EXACT format. DO NOT include any other text or greetings.
"""
        }
    ]
    return messages
def build_reasoning_eval_prompt(llm_response: str, reference_antigen: str) -> list:
    """
    Constructs a prompt for DeepSeek-R1-671B to evaluate the reasoning quality
    of a TCR antigen prediction response.

    Args:
        llm_response (str): The LLM's generated output to be evaluated.
        reference_antigen (str): The ground truth antigen sequence (for accuracy checking).

    Returns:
        list: A list of messages suitable for DeepSeek-R1-671B evaluation.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional biomedical reviewer. "
                "Given a model-generated response interpreting TCR search results, "
                "evaluate its reasoning quality based on the following four criteria:\n\n"
                "1. Format Consistency (FC): Is the format exactly as required?\n"
                "2. Information Accuracy (IA): Are all antigen mentions correct?\n"
                "3. Logical Coherence (LC): Does the justification logically support the antigen list?\n"
                "4. Medical Accuracy (MA): Is the immunological interpretation biologically plausible?\n\n"
                "Each score should be between 0 and 1. Return the scores in this exact JSON format:\n"
                "{ \"FC\": <float>, \"IA\": <float>, \"LC\": <float>, \"MA\": <float> }"
            )
        },
        {
            "role": "user",
            "content": f"""Please evaluate the following response:

Response:
{llm_response}

Ground Truth Antigen:
{reference_antigen}

Now return the scores in the required JSON format. Do not include any extra explanation or commentary."""
        }
    ]
    return messages
