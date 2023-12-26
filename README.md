# speech-emotion-recognizer

I was curious to work with speech synthesis and decided to create a simple model that would recognize the emotion of the speaker.

1. First, I looked for a SOTA speech-to-text model, potentially with an API, and found Deepgram  https://deepgram.com/.
2. Following that, I used two different emotion datasets from Kaggle, combined them, and fine-tuned a distillbert-base model on that data.
3. Then a Python code to record the speech from a microphone was written.
4. The last step was about assembling everything into one pipeline. 

Instead of using the fine-tuned model, out of curiosity I also included the possibility of asking the gpt-3.5-turbo model from OpenAI about what kind of emotion the speech includes. 
