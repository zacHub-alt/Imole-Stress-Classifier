import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
import joblib
from flask import request, session, jsonify
import requests
import os
from util import encoding_maps, encode_answers, call_llm_reasoner, final_decision, show_personalized_invite, generate_dynamic_question, UI_TRANSLATIONS
from util import questions, call_llm_reasoner_with_reasoning, user_friendly_options, user_friendly_options_pidgin, map_user_answers_to_ml, generate_dynamic_options
from utils.logging import DecisionLogger
import datetime as dt
from dotenv import load_dotenv
from supabase import create_client, Client


app = Flask(__name__)
app.secret_key = 'imole_secret_key'

load_dotenv()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE")  

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("Missing Supabase environment variables.")

# Initialize supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
def save_to_supabase(data):
    """Saves data to Supabase and returns status and response."""
    try:
        response = supabase.table(SUPABASE_TABLE).insert(data).execute()
        if response.data:
            return True, response.data
        else:
            return False, response
    except Exception as e:
        print(f"Error saving to Supabase: {e}")
        return False, str(e)


# Remove minimal Supabase insert test and debug prints

# Load ML model
MODEL_PATH = os.path.join('model', 'svm_stress_model.joblib')
model = joblib.load(MODEL_PATH)

# Initialize decision logger
decision_logger = DecisionLogger()

# Legacy static questions for fallback (multi-language, now per-feature)
QUESTIONS = {
    'pidgin': { 
        "sleep_quality": "How your sleep dey lately?",
        "appetite": "How your eating dey be like?", 
        "energy_level": "How your energy level dey right now?",
        "concentration": "You fit focus well for anything?",
        "mood_swings": "Your mood dey change?",
        "social_withdrawal": "You dey waka alone pass or you dey hang with people?",
        "negative_thoughts": "You dey get bad thoughts?",
        "academic_pressure": "School wahala dey worry you?",
        "family_conflict": "How often wahala dey for house?",
        "financial_stress": "Money palava dey",
        "religious_practice": "You dey do all those church/mosque things?"
    },
             
    'en': {
        "sleep_quality": "How would you describe your sleep quality?",
        "appetite": "How is your appetite?",
        "energy_level": "How is your energy level?",
        "concentration": "How well can you concentrate?",
        "mood_swings": "Do you experience mood swings?",
        "social_withdrawal": "Do you withdraw from social activities?",
        "negative_thoughts": "Do you have negative thoughts?",
        "academic_pressure": "How much academic pressure do you feel?",
        "family_conflict": "How often do you have family conflict?",
        "financial_stress": "How much financial stress do you have?",
        "religious_practice": "How often do you practice your religion?"
    },
    'yo': {
        "sleep_quality": "Bawo ni oorun rẹ ṣe ri?",
        "appetite": "Bawo ni ifẹkufẹ onjẹ rẹ ṣe ri?",
        "energy_level": "Bawo ni agbara rẹ ṣe wa?",
        "concentration": "Bawo ni o ṣe le dojukọ?",
        "mood_swings": "Ṣe o maa n ni iyipada iṣesi?",
        "social_withdrawal": "Ṣe o maa n yago fun awọn iṣẹ awujọ?",
        "negative_thoughts": "Ṣe o maa n ni ero buburu?",
        "academic_pressure": "Bawo ni titẹ ẹkọ ṣe wa lara rẹ?",
        "family_conflict": "Bawo ni igbagbogbo ti wahala idile ti n waye?",
        "financial_stress": "Bawo ni wahala owo ṣe wa lara rẹ?",
        "religious_practice": "Bawo ni igbagbogbo ti o n ṣe iṣe ẹsin rẹ?"
    },
    'ig': {
        "sleep_quality": "Kedu ka i si ehi ura n'abalị?",
        "appetite": "Kedu ka agụụ gị si dị?",
        "energy_level": "Kedu ka ike gị si dị?",
        "concentration": "Kedu ka i si enwe ike ilekwasị anya?",
        "mood_swings": "Ị na-enwe mgbanwe mmetụta?",
        "social_withdrawal": "Ị na-apụ n'ịbụ akụkụ nke ndị ọzọ?",
        "negative_thoughts": "Ị na-enwe echiche na-adịghị mma?",
        "academic_pressure": "Kedu ka nrụgide ọmụmụ si metụta gị?",
        "family_conflict": "Kedu ugboro ole ka esemokwu ezinụlọ na-eme?",
        "financial_stress": "Kedu ka nrụgide ego si metụta gị?",
        "religious_practice": "Kedu ugboro ole ka i na-eme omume okpukpe gị?"
    },
    'ha': {
        "sleep_quality": "Yaya barcin ka yake da daddare?",
        "appetite": "Yaya sha'awar abincin ka take?",
        "energy_level": "Yaya ƙarfin jikinka yake?",
        "concentration": "Yaya kake iya mai da hankali?",
        "mood_swings": "Kana fuskantar sauyin hali?",
        "social_withdrawal": "Kana kaucewa shiga cikin jama'a?",
        "negative_thoughts": "Kana da tunani marasa kyau?",
        "academic_pressure": "Yaya matsin lamba na karatu ke shafarka?",
        "family_conflict": "Sau nawa rikici ke faruwa a cikin iyali?",
        "financial_stress": "Yaya damuwar kudi ke shafarka?",
        "religious_practice": "Sau nawa kake gudanar da addininka?"
    }
}

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/set_country', methods=['POST'])
def set_country():
    data = request.get_json()
    if data and 'country' in data:
        session['country'] = data['country']
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error'}), 400

@app.route('/test', methods=['GET', 'POST'])
def test():
    # If first GET, show DOB prompt only
    if request.method == 'GET' and not request.args.get('language'):
        from datetime import datetime
        today_str = datetime.today().strftime('%Y-%m-%d')
        session.pop('dob', None)  # Clear any previous DOB in session
        return render_template('index.html', features=[], llm_questions={}, options_map={}, answers=None, language=None, ui=UI_TRANSLATIONS['en'], show_age_prompt=False, show_dob_prompt=True, today_str=today_str, dob=None)

    # If POST and DOB is present but language is not, show language selection (preserve DOB)
    if request.method == 'POST' and request.form.get('dob') and not request.form.get('language'):
        from datetime import datetime
        today_str = datetime.today().strftime('%Y-%m-%d')
        dob = request.form.get('dob')
        session['dob'] = dob  # Store DOB in session
        session['green_ai_banner_dismissed'] = True  # Hide Green AI banner after DOB step
        return render_template('index.html', features=[], llm_questions={}, options_map={}, answers=None, language=None, ui=UI_TRANSLATIONS['en'], show_age_prompt=False, show_dob_prompt=False, today_str=today_str, dob=dob)

    # If POST and language is selected (even if dob is blank), proceed to survey
    language = request.form.get('language') if request.method == 'POST' else None
    # Always get dob from session if available
    dob = session.get('dob')
    # If language is selected and it's not English, skip DOB logic and go straight to survey
    if language in ['pidgin', 'yo', 'ig', 'ha']:
        ui = UI_TRANSLATIONS.get(language, UI_TRANSLATIONS['en'])
        from datetime import datetime
        today_str = datetime.today().strftime('%Y-%m-%d')
        features = list(questions.keys())
        llm_questions = {f: QUESTIONS[language][f] for f in features}
        options_map = {f: questions[f][language] for f in features}
        answers = None

    # Ensure features, llm_questions, options_map are always defined before use
    features = list(questions.keys())
    llm_questions = {}
    options_map = {}
    answers = None

    def is_likely_yoruba(text):
        return any(c in text for c in ['ẹ', 'ṣ', 'ọ', 'ń', 'á', 'è', 'í', 'ò', 'ú', 'Bawo', 'ṣe', 'ni', 'wa'])
    def is_likely_igbo(text):
        return any(c in text for c in ["ị", "ọ", "ụ", "ń", "á", "è", "í", "ò", "ú", 'Kedu', 'si', 'na', 'eme'])
    def is_likely_hausa(text):
        return any(c in text for c in ["ƙ", "ɗ", "ɓ", "’", "ā", "ī", "ū", 'Yaya', 'ke', 'da', 'ne'])

    if language:
        ui = UI_TRANSLATIONS.get(language, UI_TRANSLATIONS['en'])
        from datetime import datetime
        today_str = datetime.today().strftime('%Y-%m-%d')
        show_dob_prompt = language == 'en' and dob is None and not (request.method == 'POST' and request.form.get('language'))
        english_style = 'default'
        age = None
        if language == 'en' and dob:
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d')
                today = datetime.today()
                age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                if age < 13:
                    english_style = 'genalpha'
                elif age < 30:
                    english_style = 'genz'
                else:
                    english_style = 'default'
            except Exception:
                english_style = 'default'
        for f in features:
            if language in [ 'yo', 'ig', 'ha']:
                llm_questions[f] = QUESTIONS[language][f]
                options_map[f] = questions[f][language]
            elif language in ['pidgin']:
                llm_questions[f] = QUESTIONS[language][f]
                options_map[f] = questions[f][language]
                lang_options = questions[f]['pidgin']
                options_map[f] = user_friendly_options_pidgin.get(f, lang_options)
            else:
                lang_options = questions[f]['en']
                try:
                    llm_q = generate_dynamic_question(f, lang_options, language=language, style=english_style)
                    if llm_q and llm_q.strip():
                        llm_questions[f] = llm_q
                    else:
                        llm_questions[f] = QUESTIONS['en'][f]
                except Exception:
                    llm_questions[f] = QUESTIONS['en'][f]
                # Use LLM-reworded options for Gen Z/Alpha, else user-friendly options
                if english_style in ['genz', 'genalpha']:
                    llm_opts = generate_dynamic_options(f, user_friendly_options.get(f, lang_options), language=language, style=english_style)
                    options_map[f] = llm_opts
                else:
                    options_map[f] = user_friendly_options.get(f, lang_options)
        # Store llm_options in session for mapping on POST
        if language == 'en' and english_style in ['genz', 'genalpha']:
            session['llm_options'] = options_map.copy()
        else:
            session.pop('llm_options', None)
        if request.method == 'POST' and 'language' in request.form and len(request.form) > 1 and not show_dob_prompt:
            answers = {feature: request.form.get(feature, '') for feature in features}
            print('DEBUG: language:', language, 'answers:', answers)  # Debug print
            if any(v == '' for v in answers.values()):
                incomplete_msg = {
                    'pidgin': 'Abeg answer all questions',
                    'en': 'Please answer all questions.',
                    'yo': 'Jọwọ dahun gbogbo awọn ibeere.',
                    'ig': 'Biko zaa ajụjụ niile.',
                    'ha': 'Da fatan za a amsa dukkan tambayoyin.'
                }.get(language, 'Please answer all questions.')
                flash(incomplete_msg, 'danger')
                return render_template('index.html', features=features, llm_questions=llm_questions, options_map=options_map, answers=answers, language=language, ui=ui, show_age_prompt=False, show_dob_prompt=show_dob_prompt, today_str=today_str)
            # Map user-friendly answers to ML options for English and Pidgin before logging/saving
            mapped_answers = answers.copy()
            llm_options = session.get('llm_options') if (language == 'en' and 'llm_options' in session) else None
            if language == 'en':
                mapped_answers = map_user_answers_to_ml(answers, llm_options=llm_options)
            else:
                from util import map_users_answers_to_ml_by_language
                mapped_answers = map_users_answers_to_ml_by_language(answers, language)
            # --- Sanitize all answers before saving/logging ---            # sanitize_answers_dict removed: mapped_answers is already static and clean
            processed_input = encode_answers(mapped_answers)
            input_df = pd.DataFrame([processed_input])
            ml_pred_class = model.predict(input_df)[0]
            ml_pred_probs = model.predict_proba(input_df)[0]
            stress_map = {0: "High", 1: "Low", 2: "Moderate"}
            ml_pred = stress_map.get(ml_pred_class, "Unknown")
            ml_confidence = float(np.max(ml_pred_probs))            # Call LLM reasoner and get both prediction and reasoning
            llm_pred, llm_reasoning = call_llm_reasoner_with_reasoning(mapped_answers, ml_pred)
            llm_confidence = 1.0 if llm_pred else 0.0
            # --- Use ensemble with contradiction/LLM verification ---
            from util import ensemble_with_llm_verification
            final_pred, llm_verification_pred, ml_on_llm_pred, contradiction_detected, llm_verification_reasoning = ensemble_with_llm_verification(
                mapped_answers, ml_pred, ml_confidence, llm_pred, llm_confidence, model=model, encode_fn=encode_answers
            )
            if llm_verification_pred is not None:
                print(f"[DEBUG][APP] LLM verification pred: {llm_verification_pred}, ML on LLm pred: {ml_on_llm_pred}")
            if llm_pred and llm_pred != ml_pred:
                decision_source = 'llm'
            else:
                decision_source = 'ml'
            # --- Save ensemble/LLM verification output to Supabase ---
            session['llm_verification_pred'] = llm_verification_pred
            session['ml_on_llm_pred'] = ml_on_llm_pred
            session['contradiction_detected'] = contradiction_detected
            session['llm_verification_reasoning'] = llm_verification_reasoning
            # Timeline info
            from datetime import datetime
            timeline = {
                'submitted_at': datetime.now().isoformat(),
                'ml_pred': ml_pred,
                'ml_confidence': ml_confidence,
                'llm_pred': llm_pred,
                'llm_reasoning': llm_reasoning,
                'final_pred': final_pred,
                'decision_source': decision_source
            }
            # Ensure dob is captured from session or form before logging
            dob_to_log = dob or request.form.get('dob')
            # Log all details for retraining and audit
            decision_logger.log_decision({
                'user_dob': dob_to_log,
                'language': language,
                'responses': mapped_answers,
                'ml_prediction': ml_pred,
                'ml_confidence': ml_confidence,
                'llm_prediction': llm_pred,
                'llm_reasoning': llm_reasoning,
                'final_decision': final_pred,
                'decision_source': decision_source,
                'decision_timeline': timeline
            })
            # Save results in session for access in /result
            session['user_responses'] = mapped_answers
            session['ml_prediction'] = ml_pred
            session['ml_confidence'] = ml_confidence
            session['llm_prediction'] = llm_pred
            session['llm_reasoning'] = llm_reasoning
            session['final_decision'] = final_pred
            session['decision_source'] = decision_source
            session['decision_timeline'] = timeline
            session['user_dob'] = dob_to_log  # <-- Ensure user_dob is in session for /result

            import logging
            logging.info(f"Logging user_dob: {dob_to_log}")
            save_response(language, list(mapped_answers.values()), final_pred, decision_source)
            # session.pop('dob', None)  # <-- Move this to after Supabase save in /result
            return redirect(url_for('result', level=final_pred, decision=decision_source, language=language))
        return render_template('index.html', features=features, llm_questions=llm_questions, options_map=options_map, answers=answers, language=language, ui=ui, show_age_prompt=False, show_dob_prompt=show_dob_prompt, today_str=today_str)
    ui = UI_TRANSLATIONS['en']
    from datetime import datetime
    today_str = datetime.today().strftime('%Y-%m-%d')
    features = list(questions.keys())
    llm_questions = {}
    options_map = {}
    answers = None
    return render_template('index.html', features=features, llm_questions=llm_questions, options_map=options_map, answers=answers, language=language, ui=ui, show_age_prompt=False, show_dob_prompt=False, today_str=today_str)

@app.route('/result')
def result():
    level = request.args.get('level', 'Moderate')
    decision = request.args.get('decision', 'ml')
    language = request.args.get('language', 'en')
    ui = UI_TRANSLATIONS.get(language, UI_TRANSLATIONS['en'])
    encouragements = {
        "Low": ui.get('encouragement_low', "You're doing well! Keep taking care of yourself."),
        "Moderate": ui.get('encouragement_moderate', "Remember to take breaks and talk to someone you trust."),
        "High": ui.get('encouragement_high', "It's okay to seek help. You're not alone. Consider reaching out for support.")
    }
    resources = show_personalized_invite(level)
    if language == 'en':
        from util import generate_personalized_result
        import re
        style = 'default'
        dob = session.get('dob')
        if dob:
            from datetime import datetime
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d')
                today = datetime.today()
                age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                if age < 13:
                    style = 'genalpha'
                elif age < 30:
                    style = 'genz'
            except Exception:
                style = 'default'
        personalized_result = generate_personalized_result(level, resources, style=style)
        encouragement = personalized_result or encouragements.get(level, '')
        # --- New: Extract shown resource URLs from LLM HTML ---
        shown_resource_urls = set()
        if personalized_result:
            shown_resource_urls = set(re.findall(r'href=[\'"]([^\'"]+)[\'"]', personalized_result))
        # --- Compute remaining resources ---
        all_resources = [
            (resources['waitlist'], ui['waitlist_link']),
            (resources['feedback'], ui['feedback_link']),
            (resources['self_care_game'], ui['self_care_game_link']),
            (resources['spotify'], ui['spotify_link']),
            (resources['breathing'], ui['breathing_link']),
            (resources['journaling'], ui['journaling_link'])
        ]
        remaining_resources = [(url, label) for url, label in all_resources if url not in shown_resource_urls]
    else:
        encouragement = encouragements.get(level, '')
        remaining_resources = []

    submission_data = {
    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    "user_dob": session.get('user_dob'),
    "language": language,
    "responses": session.get('user_responses', {}),
    "ml_prediction": session.get('ml_prediction', level),
    "ml_confidence": session.get('ml_confidence', 0.0),
    "llm_prediction": session.get('llm_prediction', 'Unknown'),
    "llm_reasoning": session.get('llm_reasoning', 'Unavailable'),
    "final_decision": level,
    "decision_source": decision,
    "llm_verification_pred": session.get('llm_verification_pred'),
    "ml_on_llm_pred": session.get('ml_on_llm_pred'),
    "contradiction_detected": session.get('contradiction_detected'),
    "llm_verification_reasoning": session.get('llm_verification_reasoning'),
    "country": session.get('country'),
    "decision_timeline": {
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ml_pred": session.get('ml_prediction', level),
        "ml_confidence": session.get('ml_confidence', 0.0),
        "llm_pred": session.get('llm_prediction', 'Unknown'),
        "llm_reasoning": session.get('llm_reasoning', 'Unavailable'),
        "final_pred": level,
        "decision_source": decision
        },
    "country": session.get('country')
    }
    
    # Save to Supabase
    status, response = save_to_supabase(submission_data)
    print("Supabase Save Status:", status)
    print("Response:", response)

    # Now safe to clear DOB from session
    session.pop('user_dob', None)
    session.pop('dob', None)
        
    return render_template('result.html',
        stress_level=level,
        encouragement=encouragement,
        resources=resources,
        decision_source=decision,
        ui=ui,
        language=language,
        remaining_resources=remaining_resources
    )
    


def save_response(language, answers, pred, decision_source=None):
    import csv
    path = os.path.join('data', 'responses.csv')
    os.makedirs('data', exist_ok=True)
    header = ['language'] + [f'q{i}' for i in range(1, 6)] + ['predicted_level', 'decision_source']
    row = [language] + answers + [pred, decision_source or 'ml']
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

@app.context_processor
def inject_supabase_config():
    return dict(
        config={
            'NEXT_PUBLIC_SUPABASE_URL': os.getenv('NEXT_PUBLIC_SUPABASE_URL'),
            'NEXT_PUBLIC_SUPABASE_ANON_KEY': os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY'),
            'SUPABASE_TABLE': os.getenv('SUPABASE_TABLE')
        }
    )

@app.route('/dismiss_green_ai_banner', methods=['POST'])
def dismiss_green_ai_banner():
    session['green_ai_banner_dismissed'] = True
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
