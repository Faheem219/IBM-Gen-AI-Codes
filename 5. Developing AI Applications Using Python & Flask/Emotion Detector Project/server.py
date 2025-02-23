"""
Flask-based Emotion Detection Application

This module provides an API endpoint to detect emotions in a given text
using the `emotion_detector` function.
"""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detection")

@app.route("/emotionDetector")
def sent_detector():
    """
    Analyze the user-provided text for emotions and return the result.
    
    Returns:
        str: A formatted string displaying the detected emotions and the dominant emotion.
    """
    text_to_detect = request.args.get("textToAnalyze")

    if not text_to_detect:
        return "Invalid text! Please provide input."

    response = emotion_detector(text_to_detect)

    if response.get("dominant_emotion") is None:
        return "Invalid text! Please try again."

    return (
        f"For the given statement, the system response is 'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, 'fear': {response['fear']}, "
        f"'joy': {response['joy']}, and 'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )

@app.route("/")
def render_index_page():
    """
    Render the main application page.
    
    Returns:
        Template: Renders the index.html file.
    """
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
