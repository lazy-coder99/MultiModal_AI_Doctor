# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#Step1a: Setup Text to Speech–TTS–model with gTTS
import os
import logging
from gtts import gTTS
import subprocess
import elevenlabs
from elevenlabs.client import ElevenLabs



def text_to_speech_with_gtts(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(text=input_text, lang=language, slow=False)
    audioobj.save(output_filepath)

    # Play audio only if not on Render
    # if not os.getenv("RENDER"):
    #     try:
    #         subprocess.run(['ffplay', '-nodisp', '-autoexit', output_filepath])
    #     except Exception as e:
    #         print(f"An error occurred while trying to play the audio: {e}")

    return output_filepath

# input_text = "Hi this is Sunny , building AI medical chatbot jo ki boht khatarnak hoga"
# text_to_speech_with_gtts(input_text,output_filepath="doctortesting.mp3")

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    api_key = os.environ.get("ELEVEN_API_KEY")
    client = ElevenLabs(api_key=api_key)

    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="SV61h9yhBg4i91KIBwdz",
        output_format="mp3_22050_32",
        model_id="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)

    # Play audio only if not on Render
    # if not os.getenv("RENDER"):
    #     try:
    #         subprocess.run(['ffplay', '-nodisp', '-autoexit', output_filepath])
    #     except Exception as e:
    #         print(f"An error occurred while trying to play the audio: {e}")

    return output_filepath


# input_text = "Hi, This is Sunny, building AI medical chatbot jo ki boht khatarnak hoga "
# text_to_speech_with_elevenlabs(input_text,output_filepath="doctortesting.mp3")

def text_to_speech_with_fallback(input_text, output_filepath="final.mp3"):
    try:
        return text_to_speech_with_elevenlabs(
            input_text=input_text,
            output_filepath=output_filepath
        )
    except Exception as e:
        logging.error(f"ElevenLabs TTS failed: {e}")
        logging.info("Falling back to gTTS TTS.")
        return text_to_speech_with_gtts(
            input_text=input_text,
            output_filepath=output_filepath
        )

#print(text_to_speech_with_fallback("Hello, this is a test.", "test.mp3"))