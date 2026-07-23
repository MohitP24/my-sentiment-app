import gradio as gr
from transformers import pipeline

# Load the AI model
model = pipeline("sentiment-analysis")


def analyze_sentiment(text):
    result = model(text)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)

    if label == "POSITIVE":
        return f"😊 POSITIVE — The AI is {score}% confident!"
    else:
        return f"😞 NEGATIVE — The AI is {score}% confident!"


app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        placeholder="Type a sentence here..."
    ),
    outputs=gr.Textbox(label="Result"),
    title="😊 Sentiment Analyzer",
    description="Type any sentence and I will tell you whether it is Positive or Negative!",
)

app.launch()