import os
from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import logging

from brain_of_doctor import encode_image, analyze_image_with_query
from voice_of_patient import transcribe_with_groq
from voice_of_doctor import text_to_speech_with_fallback

# Logging to debug when deploying
logging.basicConfig(level=logging.INFO)

text_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""

img_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""


def process_inputs(audio_filepath, image_filepath, chat_msg, mode):
    logging.info(f"Mode: {mode}")
    logging.info(f"Audio filepath: {audio_filepath}")
    logging.info(f"Chat message: {chat_msg}")

    # Get user input based on mode
    if mode == "voice":
        if not audio_filepath:
            return "No audio provided.", "", None
        user_input = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
    else:
        if not chat_msg:
            return "No text provided.", "", None
        user_input = chat_msg

    # Prepare image
    encoded_img = encode_image(image_filepath) if image_filepath else None
    system_prompt = img_prompt if image_filepath else text_prompt

    # Generate doctor's response
    doctor_response = analyze_image_with_query(
        query=user_input,
        encoded_image=encoded_img,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        base_prompt=system_prompt
    )

    # Generate voice output
    voice_of_doctor = text_to_speech_with_fallback(
        input_text=doctor_response,
        output_filepath="final.mp3"
    )

    return user_input, doctor_response, voice_of_doctor


# ============================ Gradio UI ============================

with gr.Blocks(title="AI Doctor with Vision, Voice, and Chat") as demo:
    mode = gr.Radio(["voice", "chat"], label="Choose your input mode:")

    # Shared image input
    image_input = gr.Image(type="filepath", label="Upload Image (Optional)")

    # Input sections
    voice_audio = gr.Audio(sources=["microphone"], type="filepath", label="Record Your Voice")
    chat_textbox = gr.Textbox(label="Type your message")

    # Visibility logic
    def toggle_mode_ui(m):
        return (
            gr.update(visible=(m == "voice")),
            gr.update(visible=(m == "chat"))
        )

    mode.change(
        toggle_mode_ui,
        inputs=mode,
        outputs=[voice_audio, chat_textbox]
    )

    # Outputs
    speech_to_text = gr.Textbox(label="Speech to Text (if voice mode)")
    doctor_response = gr.Textbox(label="Doctor's Response")
    response_audio = gr.Audio(label="Doctor's Voice")

    btn = gr.Button("Submit")

    btn.click(
        fn=process_inputs,
        inputs=[voice_audio, image_input, chat_textbox, mode],
        outputs=[speech_to_text, doctor_response, response_audio]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
