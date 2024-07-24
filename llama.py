import sys
import ollama
import numpy as np
import sounddevice as sd
import threading
import queue
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Custom prompt for Phoenix
SYSTEM_PROMPT = """You are an AI assistant named Phoenix, inspired by Jarvis from Iron Man. You should respond in a helpful, concise, and slightly witty manner. Your responses should be informative but brief, suitable for verbal communication."""

# Audio settings
RATE = 16000
CHANNELS = 1
DTYPE = np.int16

# Queue to store audio data
q = queue.Queue()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def get_ai_response(prompt):
    try:
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Sorry, I couldn't process that."

def process_audio():
    while True:
        audio_data = q.get()
        # Here you would typically process the audio data
        # For simplicity, we're just checking if there's significant audio
        if np.abs(audio_data).mean() > 500:  # Adjust this threshold as needed
            print("Audio detected, processing...")
            # Here you would typically convert audio to text
            # For this example, we'll just use a placeholder text
            user_input = "What's the weather like today?"
            print(f"User (simulated): {user_input}")
            ai_response = get_ai_response(user_input)
            print(f"Phoenix: {ai_response}")
            speak(ai_response)

def main():
    speak("Phoenix initialized. I'm listening.")
    
    with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
        print("Listening for audio...")
        threading.Thread(target=process_audio, daemon=True).start()
        input()  # Keep the main thread alive

if __name__ == "__main__":
    main()
