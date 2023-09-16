from transformers import AutoModelForSequenceClassification

"""
Command to upload: huggingface-cli login

"""

repo = "marticampgin/emotion_detectorV2"
model_name = "emotion_detectorV2"

model = AutoModelForSequenceClassification.from_pretrained("saved_model/")
model.push_to_hub(model_name)