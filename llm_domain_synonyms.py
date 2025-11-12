import json
import os
import subprocess

def query_local_llm(prompt: str, model: str = "llama3") -> str:
    """
    Query a local LLM like Ollama (llama3, mistral, etc.)
    Make sure Ollama is running locally.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        print(f"[Error] Local LLM query failed: {e}")
        return ""

def generate_domain_synonyms(domain: str, model: str = "llama3"):
    """
    Use a local LLM to generate academic/domain synonyms for key verbs/adjectives.
    Save them into a JSON cache for later use in text_cleaner.
    """
    prompt = f"""
You are an expert academic writer in the field of {domain}.
For each of the following words, provide 2-3 advanced academic synonyms commonly used in {domain} research writing.
Return your response as a JSON object mapping each word to a list of synonyms.

Words: analyze, improve, develop, design, build, optimize, evaluate, model, test, result, increase, reduce, demonstrate, support, implement

Output format:
{{
  "word1": ["syn1", "syn2"],
  "word2": ["syn1", "syn2"]
}}
"""

    print(f"🧠 Querying local LLM for domain synonyms: {domain}...")
    response = query_local_llm(prompt, model=model)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        print("[Warning] LLM response is not valid JSON. Attempting cleanup...")
        # Try to extract JSON substring
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > 0:
            try:
                data = json.loads(response[start:end])
            except Exception:
                print("[Error] Could not parse JSON.")
                return
        else:
            print("[Error] No valid JSON found in LLM response.")
            return

    cache_path = "domain_synonyms.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    cache_key = domain.lower().replace(" ", "_")
    cache[cache_key] = data

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"✅ Domain synonyms for '{domain}' saved to {cache_path}")
    return data


if __name__ == "__main__":
    # Example usage
    domain = input("Enter domain (e.g. Mechanical Engineering, Data Science, Psychology): ")
    synonyms = generate_domain_synonyms(domain)
    print(json.dumps(synonyms, indent=2))
