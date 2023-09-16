import openai

class OpenaiEmotionDetector:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.prompt_role = """You are an emotion detector. Your task is to analyze the given text, and
                             and write which emotion does the text express. Consider one of the six emotions:
                             anger, joy, sadness, disgust, surprise, fear. Choose the most likely emotion
                             and output it. You should respect the following format:

                             text: My friend haven't been feeling well lately. I am worried for him.
                             emotion: fear"""


    def ask_model(self, messages):
        # Wrapper function
        response = openai.ChatCompletion.create(
        model = self.model, messages=messages
        )
        return response["choices"][0]["message"]["content"]


    # Defining the main function
    def detect_emotion(self, text):
        prompt = f"{self.prompt_role} \
                   text: {text} \
                   emotion:"
        return self.ask_model([{"role": "user", "content": prompt}])

