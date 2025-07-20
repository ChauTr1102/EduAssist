import os
import sys
import json
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY not found. Please set it in a .env file.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# Models to use
# For audio transcription with diarization, Gemini 1.5 Pro is excellent
AUDIO_TRANSCRIPTION_MODEL = "gemini-2.5-flash"
# For text analysis and summarization, a powerful text model is sufficient
TEXT_ANALYSIS_MODEL = "gemini-2.5-flash" # Or "gemini-pro"

# --- Main Functions ---

def convert_mp4_to_mp3(video_path: Path) -> Path:
    """Converts a video file to an MP3 audio file."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    mp3_path = video_path.with_suffix(".mp3")
    print(f"Converting '{video_path.name}' to '{mp3_path.name}'...")

    try:
        video_clip = VideoFileClip(str(video_path))
        audio_clip = video_clip.audio
        if audio_clip:
            audio_clip.write_audiofile(str(mp3_path), codec='mp3')
            audio_clip.close()
        else:
            raise ValueError("The video file has no audio track.")
        video_clip.close()
        print("Conversion successful.")
        return mp3_path
    except Exception as e:
        print(f"An error occurred during video to audio conversion: {e}")
        sys.exit(1)


def transcribe_audio_with_diarization(audio_path: Path):
    """
    Transcribes an audio file and identifies different speakers.
    For long files, this will take a significant amount of time.
    """
    print(f"Uploading '{audio_path.name}' to Google AI... (This may take a while for large files)")
    audio_file = genai.upload_file(path=audio_path)
    print(f"File uploaded successfully. Starting transcription with speaker identification...")

    model = genai.GenerativeModel(model_name=AUDIO_TRANSCRIPTION_MODEL)

    # Use a tqdm progress bar for the long-running generation
    with tqdm(total=1, desc="Transcribing audio") as pbar:
        try:
            # The prompt guides the model to perform the specific task
            response = model.generate_content(
                [
                    "Transcribe this audio recording. Also, identify and label each speaker in the format 'Speaker X:'.",
                    audio_file
                ],
                request_options={"timeout": 1200} # 20 minutes timeout for very long videos
            )
            pbar.update(1)
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            # Clean up the uploaded file on error
            genai.delete_file(audio_file.name)
            sys.exit(1)

    # Clean up the uploaded file after use
    genai.delete_file(audio_file.name)
    print("Transcription complete.")
    
    # The response for Gemini 1.5 Pro with this prompt is typically a clean text string.
    # If the API returned a more structured format, you would parse it here.
    return response.text


def divide_into_sections(full_transcript: str) -> list:
    """Uses Gemini to divide the transcript into thematic sections."""
    print("Analyzing transcript to identify thematic sections...")

    model = genai.GenerativeModel(
        model_name=TEXT_ANALYSIS_MODEL,
        generation_config={"response_mime_type": "application/json"} # Enforce JSON output
    )

    prompt = f"""
    You are a text analysis expert. Your task is to analyze the following conversation transcript and divide it into logical, thematic sections based on the topics being discussed.

    Return the output as a JSON array of objects. Each object should have two keys:
    1. "title": A short, descriptive title for the section (e.g., "Introduction and Project Goals").
    2. "content": The full transcript text for that specific section.

    Ensure that the content from all sections, when combined, forms the complete original transcript.

    Here is the transcript:
    ---
    {full_transcript}
    ---
    """

    with tqdm(total=1, desc="Dividing into sections") as pbar:
        response = model.generate_content(prompt)
        pbar.update(1)
    
    try:
        # The response.text will be a JSON string, so we parse it
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error: Could not parse the sectioning response from the API. {e}")
        print("API Response was:", response.text)
        return []


def summarize_text(text: str, is_overall_summary=False) -> str:
    """Summarizes a given piece of text using Gemini."""
    model = genai.GenerativeModel(model_name=TEXT_ANALYSIS_MODEL)

    if is_overall_summary:
        task_prompt = "Provide a comprehensive summary of the entire conversation. Start with a one-paragraph executive summary, followed by a bulleted list of the main topics, key decisions, and any action items discussed."
    else:
        task_prompt = "Summarize the key points, discussions, and conclusions from the following text segment of a longer conversation. Be concise and clear."

    prompt = f"{task_prompt}\n\nHere is the text:\n---\n{text}\n---"
    
    # No progress bar here as summarization is usually fast
    response = model.generate_content(prompt)
    return response.text


def main():
    """The main orchestration function."""
    # --- 1. Get Input Video ---
    video_path_str = "H:\test1.mp4"
    video_path = Path(video_path_str.strip().strip("'\"")) # Clean up path

    if not video_path.is_file() or video_path.suffix.lower() != '.mp4':
        print(f"Error: The provided path is not a valid .mp4 file: {video_path}")
        return

    # --- 2. Convert to MP3 ---
    mp3_path = convert_mp4_to_mp3(video_path)

    # --- 3. Speech-to-Text with Speaker Diarization ---
    full_transcript = transcribe_audio_with_diarization(mp3_path)

    # Save the raw transcript
    transcript_file = video_path.with_suffix('.txt')
    transcript_file.write_text(full_transcript, encoding='utf-8')
    print(f"\nFull transcript saved to: {transcript_file}")

    # --- 4. Divide into Sections ---
    sections = divide_into_sections(full_transcript)
    if not sections:
        print("Could not divide transcript into sections. Aborting summarization.")
        return

    # --- 5. Summarize Sections and a final Overall Summary ---
    analysis_results = {
        "overall_summary": "",
        "sections": []
    }

    print("\n--- VIDEO ANALYSIS RESULTS ---")

    # Overall Summary
    print("\n[ Generating Overall Video Summary ]")
    overall_summary = summarize_text(full_transcript, is_overall_summary=True)
    analysis_results["overall_summary"] = overall_summary
    print("\n===================================")
    print("    OVERALL VIDEO SUMMARY")
    print("===================================")
    print(overall_summary)

    # Section Summaries
    print("\n[ Generating Summaries for Each Section ]")
    for i, section in enumerate(sections):
        print(f"\n--- Summarizing Section {i+1}: {section['title']} ---")
        section_summary = summarize_text(section['content'])
        
        # Store results
        analysis_results["sections"].append({
            "title": section['title'],
            "summary": section_summary,
            "full_content": section['content']
        })

        # Print results
        print(f"Title: {section['title']}")
        print("Summary:")
        print(section_summary)

    # --- 6. Save Final Results ---
    output_json_path = video_path.with_suffix('.analysis.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4)
    
    print("\n===================================")
    print(f"All analysis results have been saved to: {output_json_path}")
    print("Process finished successfully.")


if __name__ == "__main__":
    main()