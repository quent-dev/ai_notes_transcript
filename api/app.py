import speech_recognition as sr
from supabase import create_client
import os
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from dotenv import load_dotenv
import logging

# Construct the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

# Load additional libraries based on API
load_dotenv(dotenv_path, override=True)
ACTIVE_API = os.getenv("ACTIVE_API")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if ACTIVE_API == "OpenAI":
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
elif ACTIVE_API == "HuggingFace":
    import requests
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/gpt2"
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
elif ACTIVE_API == "AI21":
    import requests
    AI21_API_URL = "https://api.ai21.com/studio/v1/chat/completions"
    AI21_API_KEY = os.getenv("AI21_API_KEY")
    cleaned_str = os.getenv("AI21_COST_PER_TOKEN_INPUT").strip('[]').split(',')
    API_COST_PER_TOKEN_INPUT = [float(cost) for cost in cleaned_str]
    cleaned_str = os.getenv("AI21_COST_PER_TOKEN_OUTPUT").strip('[]').split(',')
    API_COST_PER_TOKEN_OUTPUT = [float(cost) for cost in cleaned_str]

    headers_ai21 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI21_API_KEY}"
    }
else:
    raise ValueError("Invalid ACTIVE_API specified in .env")

# Initialize the Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    transcription = request.form.get('transcription')
    if not transcription:
        logger.error('No transcription provided.')
        return jsonify({'error': 'No transcription provided.'}), 400
    
    prompt = f"Create a concise note based on the following transcription:\n\n{transcription}"
    
    payload = {
        "model": "jamba-1.5-mini",  # or "jamba-1.5-large"
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that creates concise notes from transcriptions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        logger.info("Sending transcription to AI21")
        response = requests.post(AI21_API_URL, headers=headers_ai21, json=payload)
        response.raise_for_status()
        data = response.json()
        
        note = data['choices'][0]['message']['content'].strip()
        tokens_used_input = data['usage']['prompt_tokens']
        tokens_used_output = data['usage']['completion_tokens']
        cost = calculate_cost(tokens_used_input, tokens_used_output, "exact", "jamba-1.5-mini")
        note_with_cost = f"{note}\n\n---\n**Computing Cost:** ${cost}"
        
        # Save the note_with_cost to Supabase
        try:
            logger.info("Saving note to Supabase")
            insert_response = supabase.table('notes').insert({
            "content": note,
            "cost": cost
            }).execute()
            logger.info("Note saved successfully in Supabase")
            return jsonify({
                'note': note,
                'cost': cost,
                'message': 'Note saved successfully in Supabase'
            })
        except Exception as err:
            logger.error(f"An error occurred: {err}")
            return jsonify({'error': f"An error occurred: {err}"}), 500 
    
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - {response.text}")
        return jsonify({'error': f"HTTP error occurred: {http_err} - {response.text}"}), 500
    except Exception as err:
        logger.error(f"An error occurred: {err}")
        return jsonify({'error': f"An error occurred: {err}"}), 500

@app.route('/save_note', methods=['POST'])
def save_note():
    note = request.form.get('note')
    # Insert the note into the Supabase database
    logger.info("Saving note to Supabase")
    data = supabase.table('notes').insert({"content": note}).execute()
    
    return "Note saved successfully!"

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Speak now.")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError:
        print("Sorry, there was an error with the speech recognition service.")
        return None
    
def calculate_cost(input_tokens, output_tokens, method, model):
    if model == "jamba-1.5-mini":
        cost_per_token_input = API_COST_PER_TOKEN_INPUT[0]
        cost_per_token_output = API_COST_PER_TOKEN_OUTPUT[0]
    elif model == "jamba-1.5-large":
        cost_per_token_input = API_COST_PER_TOKEN_INPUT[1]
        cost_per_token_output = API_COST_PER_TOKEN_OUTPUT[1]
    if method == "exact":
        cost = ((input_tokens * cost_per_token_input) + (output_tokens * cost_per_token_output)) / 1000000
    elif method == "estimate":
        # Estimate tokens: Simple approximation (characters / 6)
        input_tokens = len(input_tokens) / 6
        output_tokens = len(output_tokens) / 6
        cost = ((input_tokens * cost_per_token_input) + (output_tokens * cost_per_token_output)) / 1000000
    return round(cost, 6)  # Rounded to 6 decimal places

def generate_note(transcription):
    prompt = f"Create a concise note based on the following transcription:\n\n{transcription}"
    
    if ACTIVE_API == "OpenAI":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that creates concise notes from transcriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        note = response.choices[0].message['content'].strip()
    
    elif ACTIVE_API == "HuggingFace":
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7
            }
        }
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        if isinstance(response.json(), dict) and response.json().get("error"):
            note = f"Error: {response.json()['error']}"
        else:
            note = response.json()[0]['generated_text']
    
    elif ACTIVE_API == "AI21":
        MODEL = "jamba-1.5-mini"  # or "jamba-1.5-large"
        payload = {
            "model": MODEL,  # or "jamba-1.5-large"
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant that creates concise notes from transcriptions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "stream": False  # Set to True if you want streaming responses
        }
        try:
            response = requests.post(AI21_API_URL, headers=headers_ai21, json=payload)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()
            
            # Extract the generated note
            note = data['choices'][0]['message']['content'].strip()

            # Optional: If AI21 provides token usage, extract it here
            
            tokens_used_input = data['usage']['prompt_tokens']
            tokens_used_output = data['usage']['completion_tokens']
            print(f"Tokens used: {tokens_used_input} input, {tokens_used_output} output")
            # cost = tokens_used * AI21_COST_PER_TOKEN
            
            # Since token usage might not be provided, estimate cost
            cost = calculate_cost(tokens_used_input, tokens_used_output, "exact", MODEL)

            # Append cost information to the note
            note_with_cost = f"{note}\n\n---\n**Computing Cost:** ${cost}"
            
            return note_with_cost
    
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - {response.text}")
            return "Error generating note due to HTTP error."
        except Exception as err:
            print(f"An error occurred: {err}")
            return "Error generating note due to an unexpected error."
    else:
        note = "Error: No valid API selected."
    
    return note

# def save_note(note):
#     if not os.path.exists('notes'):
#         os.makedirs('notes')
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"notes/note_{timestamp}.txt"
    
#     with open(filename, 'w') as file:
#         file.write(note)
    
#     print(f"Note saved as {filename}")

def main():
    while True:
        # transcription = transcribe_audio()
        transcription = """
 When Jair Bolsonaro was the president, the narrative was Bolsonaro was a dictator, that he was a bad guy. 
 I know so many Brazilians from jiu jitsu. I know so many Brazilians, and they all love Bolsonaro. I was like, I am so confused about their politics over there. 
 I don't know what's going on, but Lulu was supposed to be this guy that was for the people. 
 And to hear that he is a part of this whole disinformation crackdown, alleged disinformation crackdown, is so disheartening.
"""
        if transcription:
            print("Transcription:", transcription)
            note = generate_note(transcription)
            print("Generated Note:", note)
            # save_note(note)
        
        choice = input("Press Enter to create another note or 'q' to quit: ")
        if choice.lower() == 'q':
            break

if __name__ == "__main__":
    app.run(debug=True)