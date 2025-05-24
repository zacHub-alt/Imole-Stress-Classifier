import re
import requests
import pandas as pd

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
        "ha": ["Matsakaici", "Mai kyau", "Mara kyau"]
    },
    "appetite": {
        "en": ["Increased", "Normal", "Reduced"],
        "yo": [" pọ si", " deede", " dinku"],
        "ig": ["Belatara", "Nkịtị", "Belatara ala"],
        "ha": ["Ya karu", "Na yau da kullum", "Ya ragu"]
    },
    "energy_level": {
        "en": ["High", "Low", "Moderate"],
        "yo": ["Giga", "Kekere", "Aropin"],
        "ig": ["Elu", "Ala", "Nkezi"],
        "ha": ["Mai yawa", "Kaɗan", "Matsakaici"]
    },
    "concentration": {
        "en": ["Focused", "Not at all", "Sometimes"],
        "yo": ["Idojukọ", "Rara", "Nigbakan"],
        "ig": ["Lekwasị anya", "Mba", "Mgbe ụfọdụ"],
        "ha": ["Mai da hankali", "A'a", "Wani lokaci"]
    },
    "mood_swings": {
        "en": ["yes", "no"],
        "yo": ["bẹẹni", "rara"],
        "ig": ["ee", "mba"],
        "ha": ["eh", "a'a"]
    },
    "social_withdrawal": {
        "en": ["yes", "no"],
        "yo": ["bẹẹni", "rara"],
        "ig": ["ee", "mba"],
        "ha": ["eh", "a'a"]
    },
    "negative_thoughts": {
        "en": ["yes", "no"],
        "yo": ["bẹẹni", "rara"],
        "ig": ["ee", "mba"],
        "ha": ["eh", "a'a"]
    },
    "academic_pressure": {
        "en": ["Mild", "Severe", "nan"],
        "yo": ["Fifinra", "Tobi", "ko si"],
        "ig": ["Nfechaa", "Ike ukwuu", "enweghị"],
        "ha": ["Mai sauƙi", "Mai tsanani", "babu"]
    },
    "family_conflict": {
        "en": ["No", "Often", "Sometimes"],
        "yo": ["Rara", "Nigbagbogbo", "Nigbakan"],
        "ig": ["Mba", "Ugboro ugboro", "Mgbe ụfọdụ"],
        "ha": ["A'a", "A kai a kai", "Wani lokaci"]
    },
    "financial_stress": {
        "en": ["Mild", "No", "Severe"],
        "yo": ["Fifinra", "Rara", "Tobi"],
        "ig": ["Nfechaa", "Mba", "Ike ukwuu"],
        "ha": ["Mai sauƙi", "A'a", "Mai tsanani"]
    },
    "religious_practice": {
        "en": ["Daily", "Never", "Rarely", "Weekly"],
        "yo": ["Lojojumo", "Rara", "Kòpọ̀", "Lọsẹ kan"],
        "ig": ["Kwa ụbọchị", "Mba", "Mgbe ụfọdụ", "Oge izu"] ,
        "ha": ["Kullum", "A'a", "Wani lokaci", "Sau ɗaya a mako"]
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

def generate_dynamic_question(feature, options, language=None, style=None):
    # Base prompt
    base = f"You are a helpful assistant for a mental health survey."
    # Style prompt
    style_prompt = ""
    if style == 'genalpha':
        style_prompt = " Use Gen Alpha style: super short, playful, emoji-rich, and kid-friendly."
    elif style == 'genz':
        style_prompt = " Use Gen Z style: casual, fun, internet slang, and some emojis."
    elif style == 'millennial':
        style_prompt = " Use Millennial style: friendly, slightly nostalgic, relatable, and a bit witty."
    # Force English for LLM if language is None or 'en'
    lang_prompt = " Write ONLY the question text in English (no preamble, no explanation, no translation, no extra formatting, and do NOT list the options)."
    # Feature prompt
    feature_prompt = f" Ask about '{feature.replace('_', ' ')}'. The question should encourage the user to pick one of the options below."
    prompt = base + style_prompt + lang_prompt + feature_prompt
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"
    }
    import requests
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        question_text = resp_json['choices'][0]['message']['content'].strip()
        return question_text
    else:
        print(f"LLM API error: {response.status_code} {response.text}")
        return None

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
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        llm_response = resp_json['choices'][0]['message']['content'].strip()
        llm_response = llm_response.rstrip('.!?').capitalize()
        if llm_response in ["Low", "Moderate", "High"]:
            return llm_response
        else:
            print(f"Unexpected LLM response: {llm_response}")
            return None
    else:
        print(f"LLM API error: {response.status_code} {response.text}")
        return None

def call_llm_reasoner_with_reasoning(user_answers, ml_prediction):
    answers_str = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_answers.items())
    prompt = f"""
You are an expert mental health assistant helping to classify stress levels (Low, Moderate, High).

Here are the user's answers to a stress-related questionnaire:
{answers_str}

The machine learning model predicted the user's stress level as: {ml_prediction}.

First, state your reasoning in 2-3 sentences (in English, even if the survey is in another language). Then, on a new line, respond only with one word: Low, Moderate, or High.
"""
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        content = resp_json['choices'][0]['message']['content'].strip()
        # Split reasoning and prediction
        lines = content.splitlines()
        reasoning = ""
        pred = None
        for line in lines:
            if line.strip().capitalize() in ["Low", "Moderate", "High"]:
                pred = line.strip().capitalize()
                break
            reasoning += line.strip() + " "
        reasoning = reasoning.strip()
        return pred, reasoning
    else:
        print(f"LLM API error: {response.status_code} {response.text}")
        return None, None

def final_decision(ml_pred, ml_confidence, llm_pred, llm_confidence):
    stress_score_map = {"Low": 0, "Moderate": 1, "High": 2}
    inverse_map = {v: k for k, v in stress_score_map.items()}
    ml_score = stress_score_map.get(ml_pred, 1)
    llm_score = stress_score_map.get(llm_pred, 1)
    combined_score = ml_confidence * ml_score + llm_confidence * llm_score
    final_score = round(combined_score)
    final_prediction = inverse_map.get(final_score, "Moderate")
    return final_prediction

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
        "journaling_link": "Bẹrẹ iwe iranti pẹlu ìtànkan",
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

def generate_dynamic_options(feature, options, language=None, style=None):
    """Call LLM to generate answer options for a survey question."""
    import re
    base = f"You are a helpful assistant for a mental health survey."
    style_prompt = ""
    if style == 'genalpha':
        style_prompt = " Use Gen Alpha style: super short, playful, emoji-rich, and kid-friendly."
    elif style == 'genz':
        style_prompt = " Use Gen Z style: casual, fun, internet slang, and some emojis."
    elif style == 'millennial':
        style_prompt = " Use Millennial style: friendly, slightly nostalgic, relatable, and a bit witty."
    # Language prompt
    if language and language != 'en':
        lang_prompt = f" Write ONLY the answer options in {language.title()} as a Python list (no preamble, no explanation, no translation, no extra formatting, and do NOT include the question)."
    else:
        lang_prompt = " Write ONLY the answer options as a Python list (no preamble, no explanation, no translation, no extra formatting, and do NOT include the question)."
    feature_prompt = f" For the question about '{feature.replace('_', ' ')}', suggest 3-5 culturally and age-appropriate answer options."
    prompt = base + style_prompt + lang_prompt + feature_prompt
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"
    }
    import requests
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        content = resp_json['choices'][0]['message']['content'].strip()
        # Try to extract a Python list from the response using regex
        match = re.search(r'\[(.*?)\]', content, re.DOTALL)
        if match:
            list_str = '[' + match.group(1) + ']'
            try:
                options = eval(list_str)
                if isinstance(options, list):
                    return [str(opt).strip() for opt in options]
            except Exception:
                pass
        # Fallback: try eval on the whole content
        try:
            options = eval(content)
            if isinstance(options, list):
                return [str(opt).strip() for opt in options]
        except Exception:
            pass
        # Fallback: return static options
        return options if isinstance(options, list) else [content]
    else:
        print(f"LLM API error (options): {response.status_code} {response.text}")
        return options  # fallback to static options

def generate_personalized_result(encouragement_level, resources, answers=None, ml_pred=None, llm_pred=None, style=None):
    """Generate a personalized encouragement and resource section for the result page using the LLM."""
    base = "You are a supportive mental health assistant."
    style_prompt = ""
    if style == 'genalpha':
        style_prompt = " Use Gen Alpha style: playful, emoji-rich, and kid-friendly."
    elif style == 'genz':
        style_prompt = " Use Gen Z style: casual, fun, internet slang, and some emojis."
    elif style == 'millennial':
        style_prompt = " Use Millennial style: friendly, slightly nostalgic, relatable, and a bit witty."
    # Compose resource links as HTML for the LLM to use directly
    resource_html = '''<ul class="resource-list" style="margin-bottom:0.7rem;">
    <li><a class="waitlist-link" href=\"{waitlist}\" target=\"_blank\">Join the Ìmọ̀lè waitlist</a></li>
    <li><a href=\"{spotify}\" target=\"_blank\">Relax with the Spotify playlist</a></li>
    <li><a href=\"{breathing}\" target=\"_blank\">Try a breathing exercise</a></li>
    <li><a href=\"{self_care_game}\" target=\"_blank\">Try the self-care game</a></li>
    <li><a href=\"{feedback}\" target=\"_blank\">Share your thoughts</a> or <a href=\"{journaling}\" target=\"_blank\">Start a journaling prompt</a></li>
    </ul>'''.format(
        waitlist=resources.get('waitlist'),
        feedback=resources.get('feedback'),
        self_care_game=resources.get('self_care_game'),
        spotify=resources.get('spotify'),
        breathing=resources.get('breathing'),
        journaling=resources.get('journaling')
    )
    # Prompt for the LLM
    prompt = (
        f"{base}{style_prompt} Based on a stress level of '{encouragement_level}', write a short, warm, and encouraging message personalized for the user. "
        f"Then, suggest 2-3 of the following resources that would be most helpful for this stress level, and explain why. Always include the waitlist and Spotify links as the first <li> in your <ul>.\n"
        f"Resources (use these HTML links in your <ul>):\n{resource_html}\n"
        f"Respond in 2-3 sentences. The encouragement message should come before the <ul>. After the <ul>, add this explanation in a <div class='resource-explanation' style='margin-top:0.7rem;color:#555;'>These resources can help you manage your high stress level by providing immediate support, teaching relaxation techniques, and promoting self-care. The breathing exercise and self-care game can help calm your mind and body, while joining the Ìmọ̀lè waitlist can connect you with a supportive community.</div> Do not include any other formatting."
    )
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"
    }
    import requests
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=payload, headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        result_text = resp_json['choices'][0]['message']['content'].strip()
        return result_text
    else:
        print(f"LLM API error (result): {response.status_code} {response.text}")
        return None
