import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        framework="pt"
    )
    # Transcribe the audio file and return the result
    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Upload the audio file")

# Launch the Gradio app
iface.launch()