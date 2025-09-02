# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()
import uuid

# VoiceBot UI with Gradio
import os
import gradio as gr
import soundfile as sf
import tempfile

from brain_of_doctor import encode_image, analyze_image_with_query
from voice_of_patient import transcribe_with_groq
from voice_of_doctor import text_to_speech_with_fallback


# System prompts
text_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

img_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_input, image_filepath, chat_msg, mode):
    # 1. Get user input based on mode
    if mode == "voice":
        if not audio_input:
            return "No audio captured", "Please try again.", None

        if isinstance(audio_input, str):
            # Already a filepath
            audio_filepath = audio_input
        else:
            # audio_input is (sr, numpy_array)
            sr, data = audio_input
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tmpfile.name, data, sr)
            audio_filepath = tmpfile.name

        user_input = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
    else:
        user_input = chat_msg

    print("DEBUG audio_input:", type(audio_input))
    print("DEBUG audio_filepath:", audio_input if isinstance(audio_input, str) else audio_filepath)
    print("DEBUG image_filepath:", image_filepath)

    # 2. Select system prompt
    encoded_img = None
    system_prompt = img_prompt if image_filepath else text_prompt

    # 3. Prepare image data if present
    encoded_img = encode_image(image_filepath) if image_filepath else None

    # 4. Generate doctor's response
    doctor_response = analyze_image_with_query(
        query=system_prompt + user_input,
        encoded_image=encoded_img,
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # 5. Generate voice output
    output_file = os.path.join(tempfile.gettempdir(), f"doctor_{uuid.uuid4().hex}.mp3")

    voice_of_doctor = text_to_speech_with_fallback(
        input_text=doctor_response,
        output_filepath=output_file
    )
    # Ensure file exists and is not empty
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return user_input, doctor_response, None
    return user_input, doctor_response, voice_of_doctor


# Build Gradio interface
with gr.Blocks(title="AI Doctor with Vision, Voice, and Chat") as demo:
    mode = gr.Radio(["voice", "chat"], label="Choose your input mode:")

    # Shared image input
    image = gr.Image(type="filepath", label="Upload Image")

    with gr.Column(visible=True) as voice_section:
        audio = gr.Audio(sources=["microphone"], type="filepath", format="wav")
        voice_chat_message = gr.Textbox(visible=False)

    with gr.Column(visible=False) as chat_section:
        chat_message = gr.Textbox(label="Type your message")
        chat_audio = gr.Audio(visible=False)  # hidden placeholder

    # Toggle sections
    mode.change(
        lambda m: (gr.Column(visible=m == "voice"), gr.Column(visible=m == "chat")),
        inputs=mode,
        outputs=[voice_section, chat_section]
    )

    # Outputs
    speech_to_text = gr.Textbox(label="Speech to Text")
    doctor_response = gr.Textbox(label="Doctor's Response")
    response_audio = gr.Audio(label="Response Audio")

    # Submit button
    btn = gr.Button("Submit")
    btn.click(
        process_inputs,
        inputs=[audio, image, chat_message, mode],
        outputs=[speech_to_text, doctor_response, response_audio]
    )

demo.launch(
    # server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))  # works for both Render and local
)

# http://127.0.0.1:7860
