from transformers import pipeline
from speech_synthesizer import SpeechSynthesizer
from openai_emotion import OpenaiEmotionDetector

user_choosing = True
current_system = None
open_ai_system = False

# Choose custom / OpenAI model
while user_choosing:
    user_system_choice = input("Welcome to the emotion detector! \
                                Please choose which system you would  \
                                like to try: [OpenAI/Custom]")

    if user_system_choice == "OpenAI":
        current_system = OpenaiEmotionDetector()
        open_ai_system = True
        user_choosing = False
    
    elif user_system_choice == "Custom":
        model_checkpoint = "marticampgin/emotion_detectorV2"
        current_system = pipeline(model=model_checkpoint)
        user_choosing = False

synthesizer = SpeechSynthesizer() # Init synthesizer
transcription = synthesizer.produce_transcription()  # Produce transcription


if open_ai_system:
    result = current_system.detect_emotion(transcription)

else:
    result = current_system(transcription)

print(f"You are saying: {transcription}")
print(f"I am {round(result['score'] * 100, 2)}% confident, that you expressed {result['label']}")