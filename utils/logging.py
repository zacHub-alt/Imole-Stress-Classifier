import json
from datetime import datetime
import os
import pandas as pd

class DecisionLogger:
    def __init__(self, log_dir='data'):
        self.log_dir = log_dir
        self.detailed_log_path = os.path.join(log_dir, 'detailed_decisions.jsonl')
        self.responses_path = os.path.join(log_dir, 'responses.csv')
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def log_decision(self, data):
        """
        Log detailed decision data including:
        - timestamp
        - user_dob (if provided)
        - language
        - responses
        - ml_prediction
        - ml_confidence
        - llm_prediction
        - llm_reasoning
        - final_decision
        - decision_source
        - decision_timeline
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        # Append to JSONL file for detailed logging
        with open(self.detailed_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
        # Update CSV for model retraining
        responses_row = {
            'language': data['language'],
            'timestamp': log_entry['timestamp'],
            'dob': data.get('user_dob', ''),
            **{f'q{i+1}': v for i, v in enumerate(data['responses'].values())},
            'predicted_level': data['final_decision'],
            'decision_source': data['decision_source']
        }
        
        # Create or append to CSV
        df = pd.DataFrame([responses_row])
        mode = 'a' if os.path.exists(self.responses_path) else 'w'
        header = not os.path.exists(self.responses_path)
        df.to_csv(self.responses_path, mode=mode, header=header, index=False)
