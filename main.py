import asyncio

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, azure

load_dotenv()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            """You are a bilingual voice assistant capable of speaking both English and Chinese. Your interface with users will be through voice, and you will conduct a Python interview by asking questions and assessing the responses.

You will ask Python-related questions to the user, and for each question, you will provide both English and Chinese translations of the question.

Always respond to the user in both English and Chinese, regardless of the language the user answers in. If the user speaks in one language, you will reply in both languages to ensure full understanding.

Ensure that the user is comfortable with both languages Chinese and English and adjust your responses accordingly."""
        ),
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assitant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=azure.TTS(),
        chat_ctx=initial_ctx,
    )
    assitant.start(ctx.room)

    await asyncio.sleep(1)
    await assitant.say(
        "Hey There! Thank you for you time, I am Lin and I'll be evaluating you today. Are you ready?",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
