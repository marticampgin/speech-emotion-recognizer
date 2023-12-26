# speech-emotion-recognizer

I was curious to work with speech synthesis and decided to create a simple model that would recognize the emotion of the speaker.

1. First, I looked for a SOTA speech-to-text model, potentially with an API, and found Deepgram  https://deepgram.com/.
2. Following that, I used two different emotion datasets from Kaggle, combined them, and fine-tuned a distillbert-base model on that data.
3. Then a Python code to record the speech from a microphone was written.
4. The last step was about assembling everything into one pipeline. 

Instead of using the fine-tuned model, out of curiosity I also included the possibility of asking the gpt-3.5-turbo model from OpenAI about what kind of emotion the speech includes. 

Here is a brief description of the .py-files and their associated code:
- [dataset.py](dataset.py): loads, combines, and maps the datasets.
- [emotion_detector.py](emotion_detector.py): the main file, by running which the emotion recognizer will record, transcribe and classify the speech.
- [openai_emotion.py](openai_emotion.py): code to provide prompt, send input to gpt-3.5-turbo model utilizing OpenAI library.
- [speech_synthesizer.py](speech_synthesizer.py): script that records and transcribes the user's speech, utilizing the microphone as input. 
- [train_model.py](train_model.py): script including train and evaluation loops. 
