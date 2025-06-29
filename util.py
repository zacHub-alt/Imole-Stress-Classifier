import re
import requests
import pandas as pd
import time
import ast
import re
import os
from dotenv import load_dotenv
from groq import Groq  # NEW: import Groq client

load_dotenv(override=True)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(f"[DEBUG] Loaded GROQ_API_KEY: {GROQ_API_KEY}")

# In-memory cache for LLM generations (question/options)
_llm_generation_cache = {}
_llm_cache_last_cleared = time.time()  # Timestamp of last cache clear
_LLM_CACHE_CLEAR_INTERVAL = 14 * 24 * 60 * 60  # 2 weeks in seconds

def _maybe_clear_llm_cache():
    global _llm_generation_cache, _llm_cache_last_cleared
    now = time.time()
    if now - _llm_cache_last_cleared > _LLM_CACHE_CLEAR_INTERVAL:
        print(f"[LLM-CACHE] Clearing LLM generation cache (2 weeks elapsed)")
        _llm_generation_cache.clear()
        _llm_cache_last_cleared = now

def validate_responses(answers):
    """Check for missing or invalid answers."""
    if any(a == '' for a in answers):
        return False, "Please answer all questions."
    if not all(str(a).isdigit() and 1 <= int(a) <= 5 for a in answers):
        return False, "Invalid answer detected."
    return True, None

def language_check(language, answers):
    """Dummy check: In real app, use NLP. Here, just check if answers are digits."""
    return all(str(a).isdigit() for a in answers)

def ai_reasoning_layer(answers):
    """Simple logic: If answers are contradictory (e.g., all low except one high), flag as high stress."""
    nums = [int(a) for a in answers if str(a).isdigit()]
    if len(nums) < 5:
        return 2  # High stress if missing/invalid
    if max(nums) - min(nums) >= 4:
        return 2  # Contradiction: some very high, some very low
    avg = sum(nums) / len(nums)
    if avg >= 4:
        return 2  # High
    elif avg >= 2.5:
        return 1  # Moderate
    else:
        return 0  # Low

# New: Fixed questions and encoding maps
questions = {
    "sleep_quality": {
        "en": ["Average", "Good", "Poor"],
        "yo": ["Aropin", "Dara", "Buburu"],
        "ig": ["Nkezi", "Oma", "Njọ"],
        "ha": ["Matsakaici", "Mai kyau", "Mara kyau"],
        "pidgin": ["Normal", "Beta", "Bad"]
    },
    "appetite": {
        "en": ["Increased", "Normal", "Reduced"],
        "yo": [" pọ si", " deede", " dinku"],
        "ig": ["Belatara", "Nkịtị", "Belatara ala"],
        "ha": ["Ya karu", "Na yau da kullum", "Ya ragu"],
        "pidgin": ["Don dey chop pass", "Normal", "No too dey chop"]
    },
    "energy_level": {
        "en": ["High", "Low", "Moderate"],
        "yo": ["Giga", "Kekere", "Aropin"],
        "ig": ["Elu", "Ala", "Nkezi"],
        "ha": ["Mai yawa", "Kaɗan", "Matsakaici"],
        "pidgin": ["Plenty energy", "Small energy", "Just dey"]
    },
    "concentration": {
        "en": ["Focused", "Not at all", "Sometimes"],
        "yo": ["Idojukọ", "Rara", "Nigbakan"],
        "ig": ["Lekwasị anya", "Mba", "Mgbe ụfọdụ"],
        "ha": ["Mai da hankali", "A'a", "Wani lokaci"],
        "pidgin": ["I fit focus", "I no fit at all", "Sometimes sha"]
    },
    "mood_swings": {
        "en": ["yes", "no"],
        "yo": ["bẹẹni", "rara"],
        "ig": ["ee", "mba"],
        "ha": ["eh", "a'a"],
        "pidgin": ["Yes", "No"]
    },
    "social_withdrawal": {
        "en": ["yes", "no"],
        "yo": ["bẹẹni", "rara"],
        "ig": ["ee", "mba"],
        "ha": ["eh", "a'a"],
        "pidgin": ["Yes", "No"]
    },
    "negative_thoughts": {
        "en": ["yes", "no"],
        "yo": ["bẹẹni", "rara"],
        "ig": ["ee", "mba"],
        "ha": ["eh", "a'a"],
        "pidgin": ["Yes", "No"]
    },
    "academic_pressure": {
        "en": ["Mild", "Severe", "nan"],
        "yo": ["Fifinra", "Tobi", "ko si"],
        "ig": ["Nfechaa", "Ike ukwuu", "enweghị"],
        "ha": ["Mai sauƙi", "Mai tsanani", "babu"],
        "pidgin": ["Small wahala", "Plenty wahala", "Nothing"]
    },
    "family_conflict": {
        "en": ["No", "Often", "Sometimes"],
        "yo": ["Rara", "Nigbagbogbo", "Nigbakan"],
        "ig": ["Mba", "Ugboro ugboro", "Mgbe ụfọdụ"],
        "ha": ["A'a", "A kai a kai", "Wani lokaci"],
        "pidgin": ["No wahala", "E dey happen", "E dey sometimes"]
    },
    "financial_stress": {
        "en": ["Mild", "No", "Severe"],
        "yo": ["Fifinra", "Rara", "Tobi"],
        "ig": ["Nfechaa", "Mba", "Ike ukwuu"],
        "ha": ["Mai sauƙi", "A'a", "Mai tsanani"],
        "pidgin": ["Small gbege", "No stress", "Big wahala"]
    },
    "religious_practice": {
        "en": ["Daily", "Never", "Rarely", "Weekly"],
        "yo": ["Lojojumo", "Rara", "ṣọ́wọn", "Lọsẹ kan"],
        "ig": ["Kwa ụbọchị", "Mba", "Mgbe ụfọdụ", "Oge izu"],
        "ha": ["Kullum", "A'a", "Wani lokaci", "Sau ɗaya a mako"],
        "pidgin": ["Everyday", "I no dey go", "Once in a while", "Every week"]
    }
}


encoding_maps = {
    "sleep_quality": {"Average": 0, "Good": 1, "Poor": 2},
    "appetite": {"Increased": 0, "Normal": 1, "Reduced": 2},
    "energy_level": {"High": 0, "Low": 1, "Moderate": 2},
    "concentration": {"Focused": 0, "Not at all": 1, "Sometimes": 2},
    "mood_swings": {"yes": 1, "no": 0},
    "social_withdrawal": {"yes": 1, "no": 0},
    "negative_thoughts": {"yes": 1, "no": 0},
    "academic_pressure": {"Mild": 0, "Severe": 1, "nan": 2},
    "family_conflict": {"No": 0, "Often": 1, "Sometimes": 2},
    "financial_stress": {"Mild": 0, "No": 1, "Severe": 2},
    "religious_practice": {"Daily": 0, "Never": 1, "Rarely": 2, "Weekly": 3}
}

# User-friendly static options for each feature
user_friendly_options_pidgin = {
    "sleep_quality": ["I sleep well well", "Na manage", "I no sleep well"],
    "appetite": ["I dey chop pass", "E still dey normal", "I no too dey chop"],
    "energy_level": ["Plenty energy dey", "Small energy dey", "I just dey"],
    "concentration": ["I fit focus well", "Sometimes I dey lose focus", "Focus no dey at all"],
    "mood_swings": ["Yes, e dey change well well", "No, e no dey really happen"],
    "social_withdrawal": ["I just dey my own", "me dey flow with everyone oo"],
    "negative_thoughts": ["I dey think bad things more", "No ooo"],
    "academic_pressure": ["small small but i fit manage", "yes oo, e too much", "Everything dey calm"],
    "family_conflict": ["Everything calm", "Plenty wahala", "Wahala sometimes"],
    "financial_stress": ["Small gbege", "No wahala", "Plenty wahala"],
    "religious_practice": ["Everyday I dey do am", "I no dey do dat kind thing", "Once in a while", "Once every week"]
}
option_map_to_ml_pidgin = {
    "sleep_quality": {
        "I sleep well well": "Good",
        "Na manage": "Average",
        "I no sleep well": "Poor"
    },
    "appetite": {
        "I dey chop pass": "Increased",
        "E still dey normal": "Normal",
        "I no too dey chop": "Reduced"
    },
    "energy_level": {
        "Plenty energy dey": "High",
        "Small energy dey": "Moderate",
        "I just dey": "Low"
    },
    "concentration": {
        "I fit focus well": "Focused",
        "Sometimes I dey lose focus": "Sometimes",
        "Focus no dey at all": "Not at all"
    },
    "mood_swings": {
        "Yes, e dey change well well": "yes",
        "No, e no dey really happen": "no"
    },
    "social_withdrawal": {
        "I just dey my own": "yes",
        "me dey flow with everyone oo": "no"
    },
    "negative_thoughts": {
        "I dey think bad things more": "yes",
        "No ooo": "no"
    },
    "academic_pressure": {
        "small small but i fit manage": "Mild",
        "yes oo, e too much": "Severe",
        "Everything dey calm": "nan"
    },
    "family_conflict": {
        "Everything calm": "No",
        "Plenty wahala": "Often",
        "Wahala sometimes": "Sometimes"
    },
    "financial_stress": {
        "Small gbege": "Mild",
        "No wahala": "No",
        "Plenty wahala": "Severe"
    },
    "religious_practice": {
        "Everyday I dey do am": "Daily",
        "I no dey do dat kind thing": "Never",
        "Once in a while": "Rarely",
        "Once every week": "Weekly"
    }
}

user_friendly_options = {
    "sleep_quality": ["Very well", "Okay", "Not well"],
    "appetite": ["Eating more", "No change", "Eating less"],
    "energy_level": ["Lots of energy", "Some energy", "Very tired"],
    "concentration": ["Easy to focus", "Sometimes hard", "Very hard to focus"],
    "mood_swings": ["Yes, a lot", "No, not really"],
    "social_withdrawal": ["Spending less time", "No change"],
    "negative_thoughts": ["More than usual", "No change"],
    "academic_pressure": ["A little stressed", "Very stressed", "Not stressed"],
    "family_conflict": ["No arguments", "A lot of arguments", "Sometimes arguments"],
    "financial_stress": ["A little worried", "Not worried", "Very worried"],
    "religious_practice": ["Every day", "Never", "Rarely", "Once a week"]
}
# Mapping from user-friendly options to ML model options
option_map_to_ml = {
    "sleep_quality": {
        "Very well": "Good",
        "Okay": "Average",
        "Not well": "Poor"
    },
    "appetite": {
        "Eating more": "Increased",
        "No change": "Normal",
        "Eating less": "Reduced"
    },
    "energy_level": {
        "Lots of energy": "High",
        "Some energy": "Moderate",
        "Very tired": "Low"
    },
    "concentration": {
        "Easy to focus": "Focused",
        "Sometimes hard": "Sometimes",
        "Very hard to focus": "Not at all"
    },
    "mood_swings": {
        "Yes, a lot": "yes",
        "No, not really": "no"
    },
    "social_withdrawal": {
        "Spending less time": "yes",
        "No change": "no"
    },
    "negative_thoughts": {
        "More than usual": "yes",
        "No change": "no"
    },
    "academic_pressure": {
        "A little stressed": "Mild",
        "Very stressed": "Severe",
        "Not stressed": "nan"
    },
    "family_conflict": {
        "No arguments": "No",
        "A lot of arguments": "Often",
        "Sometimes arguments": "Sometimes"
    },
    "financial_stress": {
        "A little worried": "Mild",
        "Not worried": "No",
        "Very worried": "Severe"
    },
    "religious_practice": {
        "Every day": "Daily",
        "Never": "Never",
        "Rarely": "Rarely",
        "Once a week": "Weekly"
    }
}

def _llm_cache_key(kind, feature, style, language):
    return f"{kind}|{feature}|{style}|{language}"

def generate_dynamic_question(feature, options, language=None, style=None):
    _maybe_clear_llm_cache()
    # Only use LLM for English if style is exactly 'genalpha' or 'genz'
    if (language is None or language == 'en') and style not in ['genalpha', 'genz']:
        # Always use static question for default English style or any non-genz/genalpha style
        return static_questions.get(feature, "How would you answer this question?")
    key = _llm_cache_key('question', feature, style, language)
    if key in _llm_generation_cache:
        return _llm_generation_cache[key]
    # Use the static question as the base for LLM style tuning
    static_q = static_questions.get(feature, f"How would you answer this question about {feature.replace('_', ' ')}?")
    style_prompt = ""
    if style == 'genalpha':
        style_prompt = "Rewrite the following question in Gen Alpha style: super short, playful, emoji-rich, and kid-friendly. Use emojis and keep it fun."
    elif style == 'genz':
        style_prompt = "Rewrite the following question in Gen Z style: casual, fun, internet slang, and little emojis. Use short sentences and relatable language."
    else:
        style_prompt = "Rewrite the following question in a simple, friendly, and encouraging style for teens."
    prompt = f"{style_prompt}\n\nQuestion: {static_q}\n\nRespond with only the rewritten question, no preamble, no explanation, no extra formatting, and do NOT list the options. The question must be a single sentence."
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        question_text = resp_json['choices'][0]['message']['content'].strip()
        # DEBUG: Log LLM question generation for all English styles
        if language == 'en' or language is None:
            print(f"[LLM-DEBUG][QUESTION] style={style} feature={feature} -> {question_text}")
        # Truncate to first sentence if too long
        if '.' in question_text:
            question_text = question_text.split('.', 1)[0].strip() + '.'
        # Language check: fallback if not English for English styles
        if (language == 'en' or language is None) and not re.search(r'[a-zA-Z]', question_text):
            print(f"[LLM-WARN][QUESTION] Non-English output for style={style} feature={feature}, falling back to static.")
            return static_q  # fallback to static question
        _llm_generation_cache[key] = question_text
        return question_text
    elif response.status_code == 429:
        print(f"[LLM-RETRY][QUESTION] Rate limit hit, retrying after 2s...")
        time.sleep(2)
        return generate_dynamic_question(feature, options, language, style)
    else:
        print(f"LLM API error: {response.status_code} {response.text}")
        return static_q  # fallback to static question


def generate_dynamic_options(feature, options, language=None, style=None):
    """Generate and clean dynamic options for a survey question using LLM, with robust flattening and no merging for English Gen Z/Alpha only. For Gen Z/Alpha, tune the static options to the style, but always map back to ML options via option_map_to_ml."""
    _maybe_clear_llm_cache()
    # Only use LLM for English if style is exactly 'genalpha' or 'genz'
    if (language is None or language == 'en') and style not in ['genalpha', 'genz']:
        # Always use static options for default English style or any non-genz/genalpha style
        cleaned_opts = [clean_option(o) for o in (options if isinstance(options, list) else []) if clean_option(o)]
        print(f"[DEBUG][OPTIONS][STATIC-EN-DEFAULT] feature={feature} style={style} lang={language} -> {cleaned_opts}")
        return cleaned_opts
    key = _llm_cache_key('options', feature, style, language)
    if key in _llm_generation_cache:
        opts = _llm_generation_cache[key]
        cleaned_opts = [clean_option(o) for o in opts if clean_option(o)]
        if len(cleaned_opts) < 2:
            return cleaned_opts
        print(f"[DEBUG][OPTIONS][CACHE] feature={feature} style={style} lang={language} -> {cleaned_opts}")
        return cleaned_opts
    # Use the static options as the base for LLM style tuning
    static_opts = user_friendly_options.get(feature, options if isinstance(options, list) else [])
    style_prompt = ""
    if style == 'genalpha':
        style_prompt = "Rewrite the following answer options in Gen Alpha style: super short, playful, emoji-rich, and kid-friendly. Use emojis and keep it fun."
    elif style == 'genz':
        style_prompt = "Rewrite the following answer options in Gen Z style: casual, fun, internet slang, and little emojis. Use short sentences and relatable language."
    else:
        style_prompt = "Rewrite the following answer options in a simple, friendly, and encouraging style for teens."
    prompt = f"{style_prompt}\n\nOptions: {static_opts}\n\nRespond with only the rewritten options as a Python list of strings, no preamble, no explanation, no extra formatting, and do NOT include the question. Each list element must be a single, distinct option."
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        content = resp_json['choices'][0]['message']['content'].strip()
        # Remove code block markers if present
        content_clean = re.sub(r'^```[a-zA-Z]*', '', content).strip()
        content_clean = re.sub(r'```$', '', content_clean).strip()
        # Try to extract a Python list from the response using regex
        match = re.search(r'\[(.*?)\]', content_clean, re.DOTALL)
        options_list = None
        if match:
            list_str = '[' + match.group(1) + ']'
            try:
                options_list = eval(list_str)
            except Exception:
                pass
        if not options_list:
            try:
                options_list = eval(content_clean)
            except Exception:
                pass
        # --- Improved: flatten and parse if the list looks like a split stringified list ---
        if options_list and isinstance(options_list, list):
            # If the list is a single string that looks like a list, or a list of fragments, join and eval
            if all(isinstance(opt, str) for opt in options_list):
                joined = ','.join(options_list)
                try:
                    parsed = eval(joined)
                    if isinstance(parsed, list):
                        options_list = parsed
                except Exception:
                    # fallback to previous flattening
                    flat = []
                    for opt in options_list:
                        if isinstance(opt, str) and opt.strip().startswith('[') and opt.strip().endswith(']'):
                            try:
                                inner = eval(opt)
                                if isinstance(inner, list):
                                    flat.extend(inner)
                                else:
                                    flat.append(opt)
                            except Exception:
                                flat.append(opt)
                        else:
                            flat.append(opt)
                    options_list = flat
        print(f"[DEBUG][OPTIONS][LLM][TYPE] feature={feature} style={style} lang={language} -> {type(options_list)}")
        if options_list and isinstance(options_list, list):
            cleaned = [clean_option(opt) for opt in options_list if clean_option(opt)]
            if len(cleaned) < 2:
                return cleaned
            print(f"[DEBUG][OPTIONS][LLM][FINAL] feature={feature} style={style} lang={language} -> {cleaned}")
            _llm_generation_cache[key] = cleaned
            return cleaned
        cleaned_opts = [clean_option(o) for o in (static_opts if isinstance(static_opts, list) else [content]) if clean_option(o)]
        print(f"[DEBUG][OPTIONS][STATIC] feature={feature} style={style} lang={language} -> {cleaned_opts}")
        _llm_generation_cache[key] = static_opts if isinstance(static_opts, list) else [content]
        return cleaned_opts
    elif response.status_code == 429:
        import time
        time.sleep(2)
        return generate_dynamic_options(feature, options, language, style)
    else:
        cleaned_opts = [clean_option(o) for o in (static_opts if isinstance(static_opts, list) else []) if clean_option(o)]
        print(f"[DEBUG][OPTIONS][ERROR] feature={feature} style={style} lang={language} -> {cleaned_opts}")
        return cleaned_opts

def sanitize_answer_value(val):
    """Sanitize a single answer value: if it's a stringified list, extract the first string element; else return as string."""
    import ast
    if isinstance(val, list):
        return str(val[0]) if val else ''
    if isinstance(val, str):
        s = val.strip()
        # Try to parse as a Python list
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and parsed:
                for v in parsed:
                    if isinstance(v, str):
                        return v
                return str(parsed[0])
        except Exception:
            pass
        # Try to split by common delimiters if it looks like a list
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        for delim in [',', ';', '|', '\n']:
            parts = [p.strip(" ' \"") for p in s.split(delim)]
            if len(parts) > 1 and all(parts):
                return parts[0]
        # Fallback: just return as string
        return s
    return str(val)

def map_user_answers_to_ml(answers, feature=None, style=None, llm_options=None):
    """
    Map user-friendly or LLM-tuned answers to ML model options for encoding.
    If LLM-tuned options are used (Gen Z/Alpha), map back to the original ML options using option_map_to_ml.
    [DEBUG 4] print statements added to trace mapping logic.
    """
    mapped = {}
    for feat, value in answers.items():
        # [DEBUG 4] Start mapping for this feature
        print(f"[DEBUG 4][MAPPING] Feature: {feat}, User Value: {value}")
        # If llm_options is provided and this feature is present, try to map by index
        if llm_options and feat in llm_options:
            try:
                idx = llm_options[feat].index(value)
                orig_opt = user_friendly_options[feat][idx]
                ml_opt = option_map_to_ml[feat][orig_opt]
                mapped[feat] = ml_opt
                print(f"[DEBUG 4][LLM-INDEX] {feat}: LLM option '{value}' at idx {idx} -> user option '{orig_opt}' -> ML option '{ml_opt}'")
                continue
            except Exception as e:
                print(f"[DEBUG 4][LLM-INDEX-ERROR] {feat}: Could not map LLM option '{value}' by index. Error: {e}")
        # Fallback to normal mapping
        if feat in option_map_to_ml and value in option_map_to_ml[feat]:
            ml_opt = option_map_to_ml[feat][value]
            mapped[feat] = ml_opt
            print(f"[DEBUG 4][DIRECT] {feat}: User value '{value}' -> ML option '{ml_opt}'")
        else:
            mapped[feat] = value  # fallback: use as-is
            print(f"[DEBUG 4][FALLBACK] {feat}: Using value as-is: '{value}'")
    return mapped

def map_users_answers_to_ml(answers): 
    """
    Map Pidgin user answers to ML model options for saving/retraining.
    Replicates the English mapping logic: direct dictionary lookup from user-facing options to ML options.
    """
    mapped = {}
    for feat, value in answers.items():
        print(f"[DEBUG 2][MAPPING] Feature: {feat}, User Value: {value}")
        if feat in option_map_to_ml_pidgin and value in option_map_to_ml_pidgin[feat]:
            ml_opt = option_map_to_ml_pidgin[feat][value]
            mapped[feat] = ml_opt
            print(f"[DEBUG 2][DIRECT] {feat}: User value '{value}' -> ML option '{ml_opt}'")
        else:
            mapped[feat] = value  # fallback: use as-is
            print(f"[DEBUG 2][FALLBACK] {feat}: Using value as-is: '{value}'")
    return mapped
    
def encode_answers(answers):
    return {f: encoding_maps[f].get(v, 0) for f, v in answers.items()}

def call_llm_reasoner(user_answers, ml_prediction):
    answers_str = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_answers.items())
    prompt = f"""
You are an expert mental health assistant helping to classify stress levels (Low, Moderate, High).

Here are the user's answers to a stress-related questionnaire:
{answers_str}

The machine learning model predicted the user's stress level as: {ml_prediction}.

Based on the above answers, please confirm this prediction or provide your own stress level classification.

Respond only with one word: Low, Moderate, or High.
"""
    client = Groq()
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}]
        )
        llm_response = completion.choices[0].message.content.strip()
        llm_response = llm_response.rstrip('.!?').capitalize()
        if llm_response in ["Low", "Moderate", "High"]:
            return llm_response
        else:
            print(f"Unexpected LLM response: {llm_response}")
            return None
    except Exception as e:
        print(f"[LLM-REASONER-EXCEPTION] {e}")
        return None


def call_llm_reasoner_with_reasoning(user_answers, ml_prediction):
    answers_str = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_answers.items())
    prompt = f"""
You are an expert mental health assistant helping to classify stress levels (Low, Moderate, High).

Here are the user's answers to a stress-related questionnaire:
{answers_str}

The machine learning model predicted the user's stress level as: {ml_prediction}.

First, state your reasoning in 2-3 sentences (in English, even if the survey is in another language). Then, on a new line by itself, respond only with one word: Low, Moderate, or High.
"""
    client = Groq()
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content.strip()
        lines = content.splitlines()
        reasoning = ""
        pred = None
        for line in lines:
            clean = line.strip().rstrip('.!?').capitalize()
            if clean in ["Low", "Moderate", "High"]:
                pred = clean
                break
            reasoning += line.strip() + " "
        reasoning = reasoning.strip()
        if not pred:
            import re
            match = re.search(r"(Low|Moderate|High)", content, re.IGNORECASE)
            if match:
                pred = match.group(1).capitalize()
        return pred, reasoning
    except Exception as e:
        print(f"[LLM-REASONER-EXCEPTION] {e}")
        return None, None

def final_decision(ml_pred, ml_confidence, llm_pred, llm_confidence):
    """
    Combine ML and LLM predictions robustly:
    - If both agree, return that prediction.
    - If they disagree, return the one with higher confidence.
    - If confidence is equal, return the higher (more conservative) stress level.
    """
    stress_score_map = {"Low": 0, "Moderate": 1, "High": 2}
    inverse_map = {v: k for k, v in stress_score_map.items()}
    ml_score = stress_score_map.get(ml_pred, 1)
    llm_score = stress_score_map.get(llm_pred, 1)

    if ml_pred == llm_pred:
        return ml_pred
    if ml_confidence > llm_confidence:
        return ml_pred
    if llm_confidence > ml_confidence:
        return llm_pred
    # If confidence is equal, return the higher stress level
    final_score = max(ml_score, llm_score)
    return inverse_map.get(final_score, "Moderate")

def show_personalized_invite(prediction):
    waitlist_url = "https://lucida-waitlist.vercel.app/" #https://imole.app/waitlist
    feedback_form_url = "https://imole.app/feedback"
    self_care_game_url = "https://www.chegg.org/taking-care-during-stressful-times" #https://imole.app/self-care-game
    spotify_playlist_url = "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0"  # Example playlist, replace with your own
    breathing_exercise_url = "https://www.chegg.org/student-mental-health-week-2024-content/the-4-7-8-breathing-method" #https://imole.app/breathing-exercise
    journaling_prompt_url = "https://writediary.com/guide/how-to-start-journaling-a-beginners-guide-to-daily-writing/what-to-write-first-in-your-journal-50-prompts-for-beginners/" #https://imole.app/journaling-prompt
    # Generate personalized links based on prediction       
    return {
        "waitlist": waitlist_url,
        "feedback": feedback_form_url,
        "self_care_game": self_care_game_url,
        "spotify": spotify_playlist_url,
        "breathing": breathing_exercise_url,
        "journaling": journaling_prompt_url
    }

UI_TRANSLATIONS = {
    "pidgin": {
        "welcome": "Welcome to Ìmọ̀lè Stress Checker. Dis tool go help you sabi how your stress level be and give you small encouragement. Abeg answer the questions well-well. Wetin you talk no be for public, e go help us support you better.",
        "select_language": "Choose your language",
        "select_option": "Choose one...",
        "submit": "Submit am",
        "clear": "Clear everything",
        "result_title": "Your Stress Result",
        "your_stress_level": "Your stress level be:",
        "encouragement_low": "You dey do well! Keep dey take care of yourself.",
        "encouragement_moderate": "Try rest small and talk to person wey you trust.",
        "encouragement_high": "Ey dey okay make you find help. You no dey alone. Try reach out make you get support.",
        "waitlist_link": "Join Ìmọ̀lè waitlist",
        "feedback_link": "Tell us wetin you think",
        "or": "or",
        "self_care_game_link": "try one small self-care game",
        "spotify_link": "Calm body with this music",
        "breathing_link": "Try small breathing exercise",
        "journaling_link": "You fe start with just one small writing",
        "take_again": "Do am again",
        "stress_low": "Small",
        "stress_moderate": "Middle",
        "stress_high": "Plenty",
        "next": "Next",
        "back": "Back",
        "llm_error": "Sorry, the sharp-brain service no dey work for now. This result na from the main tool.",
        "recommendation_title": "Wetin we suggest for you:",
        "recommendation_intro": "Try any of these things wey fit help you:"
    },
    "en": {
        "welcome": "Welcome to the Ìmọ̀lè Stress Classifier. This tool helps you understand your current stress level and offers gentle encouragement. Please answer the questions as honestly as possible. Your responses are confidential and will help us provide better support for your well-being.",
        "select_language": "Select Language",
        "select_option": "Select...",
        "submit": "Submit",
        "clear": "Clear",
        "result_title": "Stress Level Result",
        "your_stress_level": "Your Stress Level:",
        "encouragement_low": "You're doing well! Keep taking care of yourself.",
        "encouragement_moderate": "Remember to take breaks and talk to someone you trust.",
        "encouragement_high": "It's okay to seek help. You're not alone. Consider reaching out for support.",
        "waitlist_link": "Join the Ìmọ̀lè waitlist",
        "feedback_link": "Share your thoughts",
        "or": "or",
        "self_care_game_link": "try a self-care game",
        "spotify_link": "Relax with a Spotify playlist",
        "breathing_link": "Try a breathing exercise",
        "journaling_link": "Start journaling with a prompt",
        "take_again": "Take Test Again",
        "stress_low": "Low",
        "stress_moderate": "Moderate",
        "stress_high": "High",
        "next": "Next",
        "back": "Back",
        "llm_error": "Sorry, the advanced reasoning service is temporarily unavailable. Your result is based on the main model.",
        "recommendation_title": "Recommended for you:",
        "recommendation_intro": "Why not try one of these helpful resources:"
    },
    "yo": {
        "welcome": "Kaabọ si Ìmọ̀lè Ayẹwo Ipilẹ Wahala. Ẹ̀rọ yìí n ràn ọ lọwọ lati mọ ipele wahala rẹ ati gba atilẹyin. Jọwọ dahun awọn ibeere naa ni otitọ. Idahun rẹ jẹ aṣiri ati yoo ran wa lọwọ lati pese atilẹyin to dara.",
        "select_language": "Yan Ede",
        "select_option": "Yan...",
        "submit": "Firanṣẹ",
        "clear": "Nu",
        "result_title": "Abajade Ipele Wahala",
        "your_stress_level": "Ipele Wahala Rẹ:",
        "encouragement_low": "O n ṣe daradara! Maa tọju ara rẹ.",
        "encouragement_moderate": "Ranti lati sinmi ati ba ẹnikan ti o gbẹkẹle sọrọ.",
        "encouragement_high": "O dara lati wa iranlọwọ. Iwọ nikan kii ṣe. Ronu lati kan si atilẹyin.",
        "waitlist_link": "Darapọ mọ Ìmọ̀lè",
        "feedback_link": "Pin ero rẹ",
        "or": "tabi",
        "self_care_game_link": "ṣe ere itoju ara",
        "spotify_link": "Sinmi pẹlu akojọ orin Spotify",
        "breathing_link": "Ṣe adaṣe mimi",
        "journaling_link": "Bẹrẹ iwe ìranti pẹ̀lú ìtọ́kasi kan ṣoṣo",
        "take_again": "Ṣe Idanwo Lẹẹkansi",
        "stress_low": "Kekere",
        "stress_moderate": "Aarin",
        "stress_high": "Giga",
        "next": "Tẹsiwaju",
        "back": "Pada",
        "llm_error": "Ẹ jọ̀wọ́, iṣẹ́ ìmúlò ọgbọ́n atọwọda ti ìdáhùn jinlẹ̀ kò sí lọ́wọ́lọ́wọ́. Àbájáde yín dá lórí àwùjọ àkọ́kọ́.",
        "recommendation_title": "Ìmúlò fún ọ:",
        "recommendation_intro": "Ṣe o ti gbìyànjú láti lo àwọn oríṣìíríṣìí ìmúlò wíwúlò wọ̀nyí?"
    },
    "ig": {
        "welcome": "Nnọọ na Ngwa Nnyocha Nrụgide Ìmọ̀lè. Ngwaọrụ a na-enyere gị ịmata ogo nrụgide gị ugbu a ma na-enye nkwado dị nro. Biko zaa ajụjụ ndị a n'eziokwu. Azịza gị bụ ihe nzuzo ma ga-enyere anyị aka inye nkwado ka mma.",
        "select_language": "Họrọ Asụsụ",
        "select_option": "Họrọ...",
        "submit": "Zipu",
        "clear": "Hichapụ",
        "result_title": "Nsonaazụ Ọkwa Nrụgide",
        "your_stress_level": "Ọkwa Nrụgide Gị:",
        "encouragement_low": "Ị na-eme nke ọma! Nọgidenụ na-elekọta onwe gị.",
        "encouragement_moderate": "Cheta ị zuru ike ma gwa onye ị tụkwasịrị obi okwu.",
        "encouragement_high": "Ọ dị mma ịchọ enyemaka. Ị nweghị naanị. Gbalịsie ike ịkpọtụrụ nkwado.",
        "waitlist_link": "Jikọọ na Ìmọ̀lè",
        "feedback_link": "Kekọrịta echiche gị",
        "or": "ma ọ bụ",
        "self_care_game_link": "nwalee egwuregwu nlekọta onwe",
        "spotify_link": "Zuru ike na ndepụta egwu Spotify",
        "breathing_link": "Nwale mgbatị ume",
        "journaling_link": "Bido ide blọgụ na ntụziaka",
        "take_again": "Mee Nnwale Ọzọ",
        "stress_low": "Nta",
        "stress_moderate": "Etiti",
        "stress_high": "Elu",
        "next": "Na-esote",
        "back": "Laghachi",
        "llm_error": "Ndo, ọrụ ọgụgụ isi dị elu adịghị ugbu a. Nsonaazụ gị dabere na ụdị isi.",
        "recommendation_title": "Nke a ka anyị na-akwado gị:",
        "recommendation_intro": "Gbalịsie ike iji otu n’ime ihe ndị a bara uru:"
    },
    "ha": {
        "welcome": "Barka da zuwa Ìmọ̀lè Mai Tace Matsayin Damuwa. Wannan kayan aiki yana taimaka maka fahimtar matakin damuwarka kuma yana ba da kwarin gwiwa. Da fatan za a amsa tambayoyin da gaskiya. Amsoshinka sirri ne kuma za su taimaka mana mu inganta tallafi.",
        "select_language": "Zaɓi Harshe",
        "select_option": "Zaɓi...",
        "submit": "Aika",
        "clear": "Share",
        "result_title": "Sakamakon Matakin Damuwa",
        "your_stress_level": "Matakin Damuwa naka:",
        "encouragement_low": "Kana yin kyau! Ci gaba da kula da kanka.",
        "encouragement_moderate": "Ka tuna kayi hutu kuma kayi magana da wanda kake yarda da shi.",
        "encouragement_high": "Yana da kyau neman taimako. Ba kai kaɗai bane. Ka yi la'akari da neman tallafi.",
        "waitlist_link": "Shiga Ìmọ̀lè",
        "feedback_link": "Raba ra'ayinka",
        "or": "ko",
        "self_care_game_link": "gwada wasan kula da kai",
        "spotify_link": "Huta da jerin waƙoƙin Spotify",
        "breathing_link": "Gwada atisaye numfashi",
        "journaling_link": "Fara rubutun ajanda da shawarwari",
        "take_again": "Sake Gwada Test",
        "stress_low": "Ƙarami",
        "stress_moderate": "Matsakaici",
        "stress_high": "Mawadaci",
        "next": "Gaba",
        "back": "Koma baya",
        "llm_error": "Yi hakuri, sabis na basirar wucin gadi bai samu ba a yanzu. Sakamakon ku ya dogara da babban samfurin.",
        "recommendation_title": "Shawarwari a gare ku:",
        "recommendation_intro": "Me ya dace ka gwada ɗaya daga cikin waɗannan hanyoyin taimako:"
    }
}

# --- Static fallback questions for each feature (used for default English style and as fallback) ---
static_questions = {
    "sleep_quality": "How well did you sleep recently?",
    "appetite": "How has your appetite been lately?",
    "energy_level": "How much energy have you had?",
    "concentration": "How easy is it to focus on tasks?",
    "mood_swings": "Have you experienced mood swings?",
    "social_withdrawal": "Have you been spending less time with others?",
    "negative_thoughts": "Have you had more negative thoughts than usual?",
    "academic_pressure": "How stressed do you feel about school or studies?",
    "family_conflict": "Have there been arguments or conflicts at home?",
    "financial_stress": "Are you worried about money or finances?",
    "religious_practice": "How often do you participate in religious or spiritual activities?"
}

def clean_option(opt):
    """Clean a single option string: remove extra whitespace, punctuation, and normalize casing."""
    if not isinstance(opt, str):
        return str(opt)
    s = opt.strip()
    # Remove leading/trailing punctuation and whitespace
    s = re.sub(r'^[\s\-–—•\*\d\.)]+', '', s)
    s = re.sub(r'[\s\-–—•\*\d\.)]+$', '', s)
    # Normalize spaces
    s = re.sub(r'\s+', ' ', s)
    # Remove enclosing quotes
    s = s.strip('"\'')
    return s

def generate_personalized_result(level, resources, style='default'):
    """
    Use the LLM to generate a short, style-matched encouragement string (with HTML links) for teens, based on stress level and style.
    Always append the waitlist link for all users/styles/levels. Fallback to static if LLM fails.
    """
    encouragements = {
        "Low": "You're doing well! Keep taking care of yourself.",
        "Moderate": "Remember to take breaks and talk to someone you trust.",
        "High": "It's okay to seek help. You're not alone. Consider reaching out for support."
    }
    waitlist_html = f"<a href='{resources['waitlist']}' target='_blank'>Join the Ìmọ̀lè waitlist</a>"
    # Compose LLM prompt
    style_desc = {
        'genalpha': 'Gen Alpha style: playful, emoji-rich, super short, and kid-friendly. Use emojis and keep it fun.',
        'genz': 'Gen Z style: casual, fun, internet slang, and some emojis. Use short sentences and relatable language.',
        'default': 'Simple, friendly, and encouraging. Use clear, positive language for teens.'
    }.get(style, 'Simple, friendly, and encouraging. Use clear, positive language for teens.')
    resource_list = (
        f"- Self-care game: {resources['self_care_game']}\n"
        f"- Spotify playlist: {resources['spotify']}\n"
        f"- Breathing exercise: {resources['breathing']}\n"
        f"- Journaling prompt: {resources['journaling']}\n"
        f"- Feedback: {resources['feedback']}\n"
        f"- Waitlist: {resources['waitlist']}\n"
    )
    prompt = f"""
Write a short, personalized encouragement message (no more than 1- 6 sentences) for a teen who just completed a stress survey.

Start with a word or phrase of encouragement (like "Awesome!", "You got this!", "Hey, you're not alone!", etc). The user's stress level is: {level}.
Style: {style_desc}

Make it concise, positive, and easy to read for a teenager. Use the style above. Include at least one of the following helpful resource links (as HTML anchor tags):
{resource_list}

When mentioning the waitlist, present it as a way to meet new peers or connect with others like them (not just for professional help). (Do not just input the naked link but instead embedded in a text) Example: "Join the Ìmọ̀lè waitlist to meet new friends and connect with others who get it!"

Do NOT say you are an assistant, do NOT mention helping with feelings, and do NOT refer to yourself. Only encourage the user directly.

Always include the waitlist link. Respond with HTML suitable for web display (Do not just input the naked link but instead embedded in a text). Do NOT write more than 3 sentences.
"""
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            resp_json = response.json()
            llm_msg = resp_json['choices'][0]['message']['content'].strip()
            # Ensure waitlist link is present
            if resources['waitlist'] not in llm_msg:
                llm_msg += f"<br><br>{waitlist_html}"
            return llm_msg
        else:
            print(f"[LLM-ENCOURAGEMENT-ERROR] {response.status_code} {response.text}")
    except Exception as e:
        print(f"[LLM-ENCOURAGEMENT-EXCEPTION] {e}")
    # Fallback to static encouragement
    return (f"{encouragements.get(level, encouragements['Moderate'])} "
            f"<br><br>{waitlist_html}")

def map_users_answers_to_ml_yo_index(answers):
    """
    Map Yoruba user answers to canonical ML options by index using the questions dict.
    """
    mapped = {}
    for feat, value in answers.items():
        yo_opts = questions.get(feat, {}).get("yo", [])
        en_opts = questions.get(feat, {}).get("en", [])
        try:
            idx = yo_opts.index(value)
            ml_opt = en_opts[idx] if idx < len(en_opts) else value
            mapped[feat] = ml_opt
            print(f"[YO-INDEX-MAP] {feat}: '{value}' (idx {idx}) -> '{ml_opt}'")
        except Exception as e:
            mapped[feat] = value
            print(f"[YO-INDEX-MAP][FALLBACK] {feat}: '{value}' (error: {e})")
    return mapped

def map_users_answers_to_ml_ig_index(answers):
    """
    Map Igbo user answers to canonical ML options by index using the questions dict.
    """
    mapped = {}
    for feat, value in answers.items():
        ig_opts = questions.get(feat, {}).get("ig", [])
        en_opts = questions.get(feat, {}).get("en", [])
        try:
            idx = ig_opts.index(value)
            ml_opt = en_opts[idx] if idx < len(en_opts) else value
            mapped[feat] = ml_opt
            print(f"[IG-INDEX-MAP] {feat}: '{value}' (idx {idx}) -> '{ml_opt}'")
        except Exception as e:
            mapped[feat] = value
            print(f"[IG-INDEX-MAP][FALLBACK] {feat}: '{value}' (error: {e})")
    return mapped

def map_users_answers_to_ml_ha_index(answers):
    """
    Map Hausa user answers to canonical ML options by index using the questions dict.
    """
    mapped = {}
    for feat, value in answers.items():
        ha_opts = questions.get(feat, {}).get("ha", [])
        en_opts = questions.get(feat, {}).get("en", [])
        try:
            idx = ha_opts.index(value)
            ml_opt = en_opts[idx] if idx < len(en_opts) else value
            mapped[feat] = ml_opt
            print(f"[HA-INDEX-MAP] {feat}: '{value}' (idx {idx}) -> '{ml_opt}'")
        except Exception as e:
            mapped[feat] = value
            print(f"[HA-INDEX-MAP][FALLBACK] {feat}: '{value}' (error: {e})")
    return mapped

def map_users_answers_to_ml_by_language(answers, language):
    """
    General mapping dispatcher: maps user answers to canonical ML options based on language.
    Uses index-based mapping for yo, ig, ha; pidgin mapping for pidgin; English mapping for en/None.
    """
    if language == "yo":
        return map_users_answers_to_ml_yo_index(answers)
    elif language == "ig":
        return map_users_answers_to_ml_ig_index(answers)
    elif language == "ha":
        return map_users_answers_to_ml_ha_index(answers)
    elif language == "pidgin":
        return map_users_answers_to_ml(answers)
    else:
        return map_user_answers_to_ml(answers)

def ensemble_with_llm_verification(user_answers, ml_pred, ml_confidence, llm_pred, llm_confidence, model=None, encode_fn=None):
    """
    If answers are contradictory (by ai_reasoning_layer), use LLM to verify and generate its own output.
    Then, run the LLM's mapped answers through the ML model again to see if the predictions match.
    Returns: final_pred, llm_verification_pred, ml_on_llm_pred, contradiction_detected, llm_verification_reasoning
    """
    contradiction = False
    rule_pred_num = ai_reasoning_layer(list(user_answers.values()))
    num_to_label = {0: "Low", 1: "Moderate", 2: "High"}
    rule_pred = num_to_label.get(rule_pred_num, "Moderate")
    # Use encoded answers for contradiction check
    if encode_fn is not None:
        encoded = encode_fn(user_answers)
        nums = list(encoded.values())
    else:
        nums = [int(a) for a in user_answers.values() if str(a).isdigit()]
    if len(nums) >= 2 and max(nums) - min(nums) >= 2:
        contradiction = True
        print(f"[DEBUG][ENSEMBLE] Contradiction detected: nums={nums}")
    else:
        print(f"[DEBUG][ENSEMBLE] No contradiction: nums={nums}")
    if contradiction and model is not None and encode_fn is not None:
        print(f"[DEBUG][ENSEMBLE] Invoking LLM for verification...")
        llm_pred, llm_reasoning = call_llm_reasoner_with_reasoning(user_answers, ml_pred)
        print(f"[DEBUG][ENSEMBLE] LLM output: pred={llm_pred}, reasoning={llm_reasoning}")
        processed_input = encode_fn(user_answers)
        import numpy as np
        import pandas as pd
        input_df = pd.DataFrame([processed_input])
        ml_on_llm_pred_class = model.predict(input_df)[0]
        stress_map = {0: "High", 1: "Low", 2: "Moderate"}
        ml_on_llm_pred = stress_map.get(ml_on_llm_pred_class, "Unknown")
        print(f"[DEBUG][ENSEMBLE] ML prediction on LLM-verified input: {ml_on_llm_pred}")
        return llm_pred, llm_pred, ml_on_llm_pred, contradiction, llm_reasoning
    # Default: use ML/LLM ensemble as before
    print(f"[DEBUG][ENSEMBLE] Using default ensemble decision.")
    return final_decision(ml_pred, ml_confidence, llm_pred, llm_confidence), None, None, contradiction, None

def map_llm_styled_answers_to_ml(answers, llm_options, feature=None):
    """
    Map LLM-styled answers (e.g., Gen Z/Alpha) back to canonical ML options using index mapping.
    For each answer, find its index in llm_options, map to user_friendly_options, then to option_map_to_ml.
    """
    mapped = {}
    for feat, value in answers.items():
        if llm_options and feat in llm_options:
            try:
                idx = llm_options[feat].index(value)
                orig_opt = user_friendly_options[feat][idx]
                ml_opt = option_map_to_ml[feat][orig_opt]
                mapped[feat] = ml_opt
                print(f"[DEBUG][LLM-STRICT-MAP] {feat}: LLM option '{value}' at idx {idx} -> user option '{orig_opt}' -> ML option '{ml_opt}'")
                continue
            except Exception as e:
                print(f"[DEBUG][LLM-STRICT-MAP-ERROR] {feat}: Could not map LLM option '{value}' by index. Error: {e}")
        # Fallback to normal mapping
        if feat in option_map_to_ml and value in option_map_to_ml[feat]:
            ml_opt = option_map_to_ml[feat][value]
            mapped[feat] = ml_opt
            print(f"[DEBUG][LLM-STRICT-DIRECT] {feat}: User value '{value}' -> ML option '{ml_opt}'")
        else:
            mapped[feat] = value  # fallback: use as-is
            print(f"[DEBUG][LLM-STRICT-FALLBACK] {feat}: Using value as-is: '{value}'")
    return mapped



