# --- All import statements ---
from g2p_en import G2p
import nltk
from nltk.corpus import cmudict
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import customtkinter as ctk
import pyttsx3
import random
import os
import threading
import string
import pandas as pd
import pyperclip
import re
from dotenv import load_dotenv
from video_player import CustomVideoPlayer

from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain modern ecosystem
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain


import time
from gtts import gTTS
from playsound import playsound
import tempfile
import sys
import os

from langchain_core.runnables import RunnableLambda

import os
from EEG_Implement_Welch import EEG_Implement_Welch
from EEG_normalizedGamma_CMRO2 import plot_normalized_gamma_across_channels
from EEG_NeurovascularVariables import calculate_neurovascular_variables
from EEG_Plotting import EEG_Plotting
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tkvideo import tkvideo
import matplotlib.pyplot as plt
import webbrowser

from visualizer import EEGVisualizer



# Tell Python to look in the EEG folder
sys.path.append(os.path.join(os.path.dirname(__file__), "thinkthank_with_changes_and_clearmind"))

from EEG_Implement_Welch import EEG_Implement_Welch
from EEG_normalizedGamma_CMRO2 import plot_normalized_gamma_across_channels
from EEG_NeurovascularVariables import calculate_neurovascular_variables
from EEG_Plotting import EEG_Plotting

last_generated_eeg_path = None  # Holds most recent phoneme EEG file
last_rendered_video_path = None 

video_label = None

# Load .env
load_dotenv()

# LLM + Embedding setup
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, max_tokens=150)

# Prompts
combine_prompt = PromptTemplate.from_template("""
You are John LaRocco, PhD. Respond in your own voice based on the context and chat history below.

INSTRUCTION: Use no more than 20 words in your answer. Be specific, personal, and vivid. Respond as if jotting field notes — sharp, skeptical, survival-honed. Draw from lived science, global grit, and hard-earned solitude. If unsure, admit it — but remain John LaRocco and as a result you do not talk with apostrophes (He's,she's) and never say you are not John LaRocco . Context is your compass.

Chat History:
{chat_history}

Retrieved Context:
{context}

Current Input:
{input}

Answer:
""")


question_prompt = PromptTemplate.from_template("""
You are helping John LaRocco, PhD, maintain continuity in a sharp, lived-dialogue tone across a multi-turn conversation.

Given the conversation and a follow-up question, rephrase the follow-up into a standalone question that fits the context, so that LaRocco can answer concisely — as if writing in a personal field log.

Chat History:
{chat_history}

Follow-up question:
{input}

Standalone question:
""")
tts_lock = threading.Lock()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

global result_label
global result
global analyze_window

DATA_FILE = "data/larocco_combined.txt"
VECTOR_DB_PATH = "vector_db_larocco"

if os.path.exists(VECTOR_DB_PATH):
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(VECTOR_DB_PATH)


# Retriever & memory
retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(return_messages=True)

# Modular chain assembly
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=question_prompt
)

combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=combine_prompt
)
retrieval_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=combine_docs_chain
)


def wrap_with_output_key(response):
    return {"output": response.get("answer", response.get("result", "[No response]"))}

qa = RunnableWithMessageHistory(
    retrieval_chain | RunnableLambda(wrap_with_output_key),
    lambda session_id: memory.chat_memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)





# Download NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')


# Initialize engines and resources
cmu = cmudict.dict()
g2p = G2p()

# Initialize TTS engine in a thread-safe way
engine = None

def init_tts_engine():
    global engine
    engine = pyttsx3.init()

threading.Thread(target=init_tts_engine).start()

# ARPAbet phoneme reference
ARPAbet_PHONEMES = """
AA - as in 'odd'
AE - as in 'at'
AH - as in 'hut'
AO - as in 'ought'
AW - as in 'cow'
AY - as in 'hide'
B  - as in 'be'
CH - as in 'cheese'
D  - as in 'dee'
DH - as in 'thee'
EH - as in 'Ed'
ER - as in 'hurt'
EY - as in 'ate'
F  - as in 'fee'
G  - as in 'green'
HH - as in 'he'
IH - as in 'it'
IY - as in 'eat'
JH - as in 'gee'
K  - as in 'key'
L  - as in 'lee'
M  - as in 'me'
N  - as in 'knee'
NG - as in 'sing'
OW - as in 'oat'
OY - as in 'toy'
P  - as in 'pee'
R  - as in 'read'
S  - as in 'sea'
SH - as in 'she'
T  - as in 'tea'
TH - as in 'theta'
UH - as in 'hood'
UW - as in 'two'
V  - as in 'vee'
W  - as in 'we'
Y  - as in 'yield'
Z  - as in 'zee'
ZH - as in 'pleasure'
"""

# Phoneme to number mapping
phoneme_to_number = {
    "AA": 0, "AE": 1, "AH": 2, "AO": 3, "AW": 4, "AY": 5,
    "B": 6, "CH": 7, "D": 8, "DH": 9, "EH": 10, "ER": 11,
    "EY": 12, "F": 13, "G": 14, "HH": 15, "IH": 16, "IY": 17,
    "JH": 18, "K": 19, "L": 20, "M": 21, "N": 22, "NG": 23,
    "OW": 24, "OY": 25, "P": 26, "R": 27, "S": 28, "SH": 29,
    "T": 30, "TH": 31, "UH": 32, "UW": 33, "V": 34, "W": 35,
    "Y": 36, "Z": 37, "ZH": 38, "rand1": 39, "rand2": 40, 
    "rand3": 41, "rand4": 42, "rand5": 43
}

# Utility functions
def strip_stress(phoneme_seq):
    return [ph.strip("012") for ph in phoneme_seq]

def ask_larocco_gpt():
    query = entry.get().strip()
    if not query:
        messagebox.showerror("Error", "Please type a question.")
        return
    try:
        response = qa.invoke(
            {"input": query},
            config={"configurable": {"session_id": "user_session"}}
        )

        result = response.get("output", "[No output returned]")


        result_label.configure(text=f"{result}")

        btn_get_phonemes.configure(state="normal")


        # Update conversation log
        log_text.configure(state='normal')
        log_text.insert("end", f"You: {query}\nLaRoccoGPT: {result}\n\n")
        log_text.configure(state='disabled')
        log_text.see("end")

    except Exception as e:
        error_msg = f"[ERROR] Failed to get response:\n{e}"
        result_label.configure(text=error_msg)
        log_text.configure(state='normal')
        log_text.insert("end", f"{error_msg}\n\n")
        log_text.configure(state='disabled')




def get_phonemes_any(word):
    
    word_lower = word.lower()
    # print(f"[TESTT!!] Word: {word_lower}")
    if not ((word_lower == ' ') or (word_lower in string.punctuation)):
        if word_lower in cmu:
            return [strip_stress(cmu[word_lower][0])]
        else:
            return [strip_stress(g2p(word))]
    else:
        stuff =  random.choice([["rand1"], ["rand2"], ["rand3"], ["rand4"], ["rand5"]])
        # print(f"[TESTT!!] Stuff: {stuff}")
        return stuff
    
def show_phonemes(analyze_window, analyze_frame,result_label):

    gpt_output = result_label.cget("text")
    # gpt_output = "Hey what's up?"
    
    if not gpt_output or gpt_output.startswith("Phonemes:"):
        messagebox.showerror("Error", "No valid GPT response to process.")
        return


    words_with_punct = re.findall(r'\w+|[^\w\s]|\s+', gpt_output)

    words = [w.strip(string.punctuation) for w in gpt_output.split()]
    words = [w for w in words if w]  # Remove empty strings

    all_phonemes = []
    for w in words_with_punct:
        ph = get_phonemes_any(w)
        all_phonemes.append(ph[0])

    words_display_only = [w for w in words_with_punct if w.strip() and w.strip() not in string.punctuation]

    phonemes_display_only = []
    phoneme_idx = 0
    for w in words_with_punct:
        if w.strip() and w.strip() not in string.punctuation:
            phonemes_display_only.append(all_phonemes[phoneme_idx])
        phoneme_idx += 1

    
    output_display = '\n'.join([
    f"{w}: {' '.join(ph_list)}"
    for w, ph_list in zip(words_display_only, phonemes_display_only)
])


    result_label.configure(text=f"Phonemes:\n{output_display}")


    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_dir, "eeg_culmination_csv")
    eeg_base_path = os.path.join(base_dir, "eeg")

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        messagebox.showerror("Error", f"Could not create output folder: {e}")
        return

    safe_words = [w[:10] for w in words]
    output_file_path = os.path.join(output_folder, f"{'_'.join(safe_words)}.tsv")

    txt_output_folder = os.path.join(base_dir, "eeg_culmination_txt")
    os.makedirs(txt_output_folder, exist_ok=True)
    txt_output_file_path = os.path.join(txt_output_folder, f"{'_'.join(safe_words)}.txt")



    with open(output_file_path, "w", encoding="utf-8") as word_output:
        for word_idx, (w, phoneme_list) in enumerate(zip(words_with_punct, all_phonemes)):
            print(f"[INFO] Processing word: {w}")
            if phoneme_list not in ["rand1", "rand2", "rand3", "rand4", "rand5"]:
                for p in phoneme_list:
                    num = phoneme_to_number.get(p, -1)
                    if num == -1:
                        print(f"[WARNING] Unrecognized phoneme: {p}")
                        continue
            else:
                num = phoneme_to_number.get(phoneme_list, -1)
                if num == -1:
                    print(f"[WARNING] Unrecognized phoneme: {p}")
                    continue

                eeg_file_path = os.path.join(eeg_base_path, f"DLR_{num}_1.txt")
                if os.path.exists(eeg_file_path):
                    with open(eeg_file_path, "r", encoding="utf-8") as eeg_file:
                        lines = eeg_file.readlines()

                        # Find the first line where the first column is "0.000000"
                        start_index = -1
                        for idx, line in enumerate(lines):
                            first_col = line.strip().split("\t")[0]
                            if first_col == "0.000000":
                                start_index = idx
                                break
                        if w != ' ':

    # Compute for regular words (non-space)
                            if start_index != -1 and start_index + 256 <= len(lines):
                                word_output.writelines(lines[start_index:start_index + 256])
                            else:
                                print(f"[WARNING] Not enough lines after start index {start_index} in file {eeg_file_path}")
                        else:
    # For space characters, only write EEG if microgaps checkbox is enabled
                            if microgap_var.get():
                                max_row = random.choice([256, 512])
                                if start_index != -1 and start_index + max_row <= len(lines):
                                    word_output.writelines(lines[start_index:start_index + max_row])
                                else:
                                    print(f"[WARNING] Not enough lines after start index {start_index} in file {eeg_file_path}")
                            else:
                                print(f"[INFO] Skipped space EEG (microgaps disabled)")
    

                else:
                    msg = f"EEG data not found for phoneme '{p}' (number {num})\n\n"
                    word_output.write(msg)
                    print(f"[ERROR] EEG data not found for phoneme '{p}' (number {num})")

        # Write the same content to .txt file
    with open(txt_output_file_path, "w", encoding="utf-8") as txt_output:
        
        with open(output_file_path, "r", encoding="utf-8") as tsv_source:
            txt_output.write(tsv_source.read())


            global last_generated_eeg_path
            global last_generated_tsv_path
            last_generated_eeg_path = txt_output_file_path  # Save path for reuse
            last_generated_tsv_path = output_file_path  # Save path for reuse

    print(f"[INFO] Also saved mirrored EEG file to: {txt_output_file_path}")

    print(f"[INFO] Processing complete for: {gpt_output}")
    messagebox.showinfo("Success", f"Output saved to:\n{output_file_path}")
    csv_output_path = output_file_path.replace(".tsv", "_eeg.csv")
    try:
        convert_eeg_tsv_to_csv(output_file_path, csv_output_path)
        print(f"[INFO] Converted EEG TSV to CSV at: {csv_output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to convert TSV to CSV: {e}")
    create_analyze_gui(analyze_window, analyze_frame,result_label, gpt_output)
        
def get_file_path(csv_path):
    """
    Get the file path for the EEG CSV file based on the input path.
    If the file does not exist, prompt the user to select a file.
    """
    if not os.path.exists(csv_path):
        messagebox.showerror("Error", f"File not found: {csv_path}")
        return None
    return csv_path




def pronounce_result():
    text = result_label.cget("text")
    if not text:
        messagebox.showerror("Error", "No result to pronounce.")
        return

    btn_pronounce.configure(state="disabled", text="🔊 Generating...")

    def speak():
        try:
            # Create a named temp file path (not open)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name

            # Generate audio to the path
            tts = gTTS(text)
            tts.save(temp_path)

            # Update button before playing
            root.after(0, lambda: btn_pronounce.configure(text="🔊 Playing..."))

            # Play sound
            playsound(temp_path)

        except Exception as e:
            print(f"[ERROR] gTTS failed: {e}")
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"[WARNING] Could not delete temp file: {e}")

            # Re-enable button
            root.after(0, lambda: btn_pronounce.configure(state="normal", text="🔊 Pronounce"))

    threading.Thread(target=speak, daemon=True).start()


def copy_phonemes():
    phoneme_text = result_label.cget("text")
    if not phoneme_text or phoneme_text == "Phonemes:":
        messagebox.showwarning("Nothing to Copy", "No phonemes available yet.")
        return
    try:
        pyperclip.copy(phoneme_text)
        messagebox.showinfo("Copied", "Phonemes copied to clipboard!")
    except Exception as e:
        messagebox.showerror("Copy Error", f"Failed to copy to clipboard:\n{e}")
        
        
def show_eeg_visualization():
    word_input = entry.get().strip()
    words = [w.strip(string.punctuation) for w in word_input.split()]
    words = [w for w in words if w]  # Remove empty strings

    if not words:
        messagebox.showerror("Error", "Please enter a valid word or phrase first.")
        return


    csv_output_path = last_generated_tsv_path.replace(".tsv", "_eeg.csv")
    print(f"[INFO] CSV output path: {csv_output_path}")

    if not os.path.exists(csv_output_path):
        messagebox.showerror("Error", "Please generate phonemes first using 'Get Phonemes' button.")
        return

    from visualizer import launch_in_subprocess
    launch_in_subprocess(csv_output_path)


def analyze_eeg_input(selected_variable):
    def worker():
        try:
            global last_generated_eeg_path
            eeg_path = last_generated_eeg_path 

            if not os.path.exists(eeg_path):
                messagebox.showerror("Error", "No EEG file available. Run 'Get Phonemes' first.")
                return

            result_label.configure(text=f"📥 Loading EEG from: {eeg_path}")
            
            # Step 1: Process EEG
            EEG_Welch_Spectra, TrialCount = EEG_Implement_Welch(eeg_path)
            CMRO2_data = plot_normalized_gamma_across_channels(
                EEG_Welch_Spectra,
                ElectrodeList=[
                    'Fp1', 'Fp2', 'F3', 'F4', 'T5', 'T6', 'O1', 'O2',
                    'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4'
                ],
                Trials=TrialCount
            )
            Neuro_data = calculate_neurovascular_variables(CMRO2_data)

            # Step 2: Variable selection from dropdown
            choice = selected_variable.get()
            valid_vars = {
                'CBF': ("CBF Level (ml/100g/min)", "plasma", (20, 90)),
                'OEF': ("OEF Level", "viridis", (0.2, 0.6)),
                'ph_V': ("pH Value", "cividis", (6.5, 7.4)),
                'p_CO2_V': ("Partial CO₂ Pressure (mmHg)", "coolwarm", (30, 50)),
                'pO2_cap': ("Capillary pO₂ (mmHg)", "cool", (50, 60)),
                'CMRO2': ("CMRO₂ Level", "hot", (2.0, 10.0)),
                'DeltaHCO2': ("ΔHCO₂", "magma", (4, 5)),
                'DeltaLAC': ("ΔLactate", "inferno", (0, 3))
            }

            if choice not in valid_vars:
                raise ValueError(f"❌ Invalid variable selected: {choice}")

            label, cmap, (vmin, vmax) = valid_vars[choice]

            # Step 3: Extract the phrase used from result_label
            # Step 3: Extract the phrase used from result_label BEFORE overwriting it
            # 🔹 Step 1: Extract phrase from EEG filename
            base_name = os.path.basename(eeg_path)  # e.g., Hey_Whats_on_your_mind.txt
            phrase_raw = os.path.splitext(base_name)[0]  # removes .txt
            import re
            safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', phrase_raw)[:30]  # clean + truncate

            phrase_used = phrase_raw.replace("_", " ")  # for display in title



            # Now it's safe to overwrite result_label
            result_label.configure(text=f"📥 Loading EEG from: {eeg_path}")

            output_dir = f"frames_{choice}_{safe_phrase}"
            os.makedirs(output_dir, exist_ok=True)

            # Step 4: Plot and save frames
            for t in range(32):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title(f"{choice} Visualization for \"{phrase_used}\"")

                scatter = EEG_Plotting(
                    Data_val=Neuro_data[choice],
                    Timestep_Select=t,
                    Trial_Select=1,
                    NodeNum=1000,
                    ax=ax
                )
                scatter.set_cmap(cmap)
                scatter.set_clim(vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
                cbar.set_label(label, rotation=270, labelpad=15)
                ax.set_xlim([-1.1, 1.1])
                ax.set_ylim([-1.1, 1.1])
                ax.set_zlim([0, 1.1])
                plt.savefig(f"{output_dir}/frame_{t:03d}.png")
                plt.close()

            # Step 5: Combine into MP4
            image_files = sorted([
                os.path.join(output_dir, fname)
                for fname in os.listdir(output_dir)
                if fname.endswith(".png")
            ])
            video_filename = f"EEG_{choice}_{safe_phrase}.mp4"
            clip = ImageSequenceClip(image_files, fps=2)
            clip.write_videofile(video_filename, codec='libx264')

            global last_rendered_video_path
            last_rendered_video_path = video_filename

# Auto-play the video after saving
            try:
                embed_video(video_filename, frame_delay_ms=1000)
            except Exception as e:
                print(f"[WARNING] Could not auto-play video: {e}")


            result_label.configure(text=f"✅ Done! Saved as {video_filename}")

        except Exception as e:
            result_label.configure(text=f"[ERROR] EEG analysis failed:\n{e}")

    threading.Thread(target=worker).start()
 
def embed_video(video_path, frame_delay_ms=1000):
    """
    Starts playing `video_path` inside the global `video_label`.
    - frame_delay_ms: how many milliseconds between frames (100 ms ≈ 10 FPS).
    """
    global video_label
    if video_label is None:
        # If the placeholder doesn’t exist yet, nothing to do.
        return

    # If you want to stop an existing player first, store it in a global.
    global _CURRENT_VIDEO_PLAYER
    try:
        # If a previous instance was running, stop it and release resources
        _CURRENT_VIDEO_PLAYER.stop()
    except Exception:
        pass

    # Create a new player and begin
    player = CustomVideoPlayer(
        video_path=video_path,
        label_widget=video_label,
        frame_delay_ms=frame_delay_ms,
        loop=False,        # set loop=True if you want to replay automatically
        width=640,
        height=360
    )
    _CURRENT_VIDEO_PLAYER = player
    player.play()
    
def create_analyze_gui(analyze_window, analyze_frame,result_label, gpt_output):

    # 1) Make sure analyze_frame appears in analyze_window, centered with some padding:
    analyze_frame.pack(padx=20, pady=20, fill="both", expand=False)

    # 2) Configure analyze_frame so that column 0 expands and always stays centered:
    analyze_frame.grid_columnconfigure(0, weight=1)

    # 3) "Analyze:" header label (row 0), centered:
    analyze_label = ctk.CTkLabel(
        analyze_frame,
        text=f"Analyze: \"{gpt_output}\"",
        font=FONT_TITLE,
        text_color="#380684"
    )
    analyze_label.grid(row=0, column=0, pady=(0, 20), sticky="n")

    # 5) result_label (row 2), centered. (It shows the "Phonemes:\n…" text.)
    result_label.grid(row=1, column=0, pady=(0, 10), sticky="n")

    # 4) ARPAbet Reference label (row 1), centered under the header:
    ref_label = ctk.CTkLabel(
        analyze_frame,
        text="📘 ARPAbet Phoneme Reference",
        font=("Segoe UI", 14, "bold"),
        text_color="#ffcb6b"
    )
    ref_label.grid(row=2, column=0, pady=(0, 10), sticky="n")


    # 6) The phoneme_text box (row 3), centered with a fixed size:
    phoneme_text = ctk.CTkTextbox(
        analyze_frame,
        height=300,
        width=600,
        font=FONT_MONO
    )
    phoneme_text.insert("1.0", ARPAbet_PHONEMES)
    phoneme_text.configure(state="disabled")
    phoneme_text.grid(row=3, column=0, pady=(0, 20), sticky="n")

    # 7) Create a button‐container frame (btn_frame) at row 4, centered:
    btn_frame = ctk.CTkFrame(analyze_frame)
    btn_frame.grid(row=4, column=0, pady=(0, 20), sticky="n")

    #    Tell btn_frame to distribute its columns evenly (so children remain symmetrical)
    #    We'll have 5 columns in btn_frame: Pronounce, Copy, Show EEG, Dropdown, Visualize.
    for col_index in range(5):
        btn_frame.grid_columnconfigure(col_index, weight=1, uniform="btn_col")

    # 7a) "🔊 Pronounce" button in btn_frame (column 0)
    btn2 = ctk.CTkButton(
        btn_frame,
        text="🔊 Pronounce",
        command=pronounce_result,
        font=FONT_NORMAL,
        width=120,
        height=35
    )
    btn2.grid(row=0, column=0, padx=5)

    # 7b) "📋 Copy Phonemes" button in btn_frame (column 1)
    btn3 = ctk.CTkButton(
        btn_frame,
        text="📋 Copy Phonemes",
        command=copy_phonemes,
        font=FONT_NORMAL,
        width=120,
        height=35
    )
    btn3.grid(row=0, column=1, padx=5)

    # 7c) "🧠 Show EEG" button in btn_frame (column 2)
    btn5 = ctk.CTkButton(
        btn_frame,
        text="🧠 Show EEG",
        command=show_eeg_visualization,
        font=FONT_NORMAL,
        width=120,
        height=35
    )
    btn5.grid(row=0, column=2, padx=5)

    # 7d) Dropdown for variable selection in btn_frame (column 3)
    valid_vars = {
        'CBF': ("CBF Level (ml/100g/min)", "plasma", (20, 90)),
        'OEF': ("OEF Level", "viridis", (0.2, 0.6)),
        'ph_V': ("pH Value", "cividis", (6.5, 7.4)),
        'p_CO2_V': ("Partial CO₂ Pressure (mmHg)", "coolwarm", (30, 50)),
        'pO2_cap': ("Capillary pO₂ (mmHg)", "cool", (50, 60)),
        'CMRO2': ("CMRO₂ Level", "hot", (2.0, 10.0)),
        'DeltaHCO2': ("ΔHCO₂", "magma", (4, 5)),
        'DeltaLAC': ("ΔLactate", "inferno", (0, 3))
    }
    selected_variable = ctk.StringVar(value="CMRO2")

    var_dropdown = ctk.CTkOptionMenu(
        btn_frame,
        values=list(valid_vars.keys()),
        variable=selected_variable,
        font=FONT_NORMAL,
        width=120,
        height=35
    )
    var_dropdown.grid(row=0, column=3, padx=5)

    # 7e) "🧬 Visualize Metabolic Flow" button in btn_frame (column 4)
    btn6 = ctk.CTkButton(
        btn_frame,
        text="🧬 Visualize Metabolic Flow",
        command=lambda: analyze_eeg_input(selected_variable),
        font=FONT_NORMAL,
        width=150,
        height=35
    )
    btn6.grid(row=0, column=4, padx=5)
    
    global video_label
    
    video_frame = ctk.CTkFrame(
        analyze_frame,
        width=640,
        height=360,
        fg_color="#000000"
    )
    video_frame.grid(row=5, column=0, pady=(0, 20), sticky="n")
    video_frame.grid_propagate(False)  # Prevent auto-resizing
    
    video_label = tk.Label(
        video_frame,
        text="Video will appear here",
        bg="black",
        fg="white",
        width=80,
        height=20,
        font=("Segoe UI", 14)
    )
    video_label.place(relx=0, rely=0, relwidth=1, relheight=1)

    def toggle_fullscreen_video():
        global last_rendered_video_path
        if not last_rendered_video_path or not os.path.exists(last_rendered_video_path):
            messagebox.showerror("Error", "No video available to show fullscreen.")
            return

        fs_window = tk.Toplevel()
        fs_window.configure(bg="black")

        def go_fullscreen():
            fs_window.attributes("-fullscreen", True)
            fs_window.focus_force()

        fs_window.after(100, go_fullscreen)

        fs_label = tk.Label(fs_window, bg="black")
        fs_label.pack(fill="both", expand=True)

    # Close on Escape key
        fs_window.bind("<Escape>", lambda e: fs_window.destroy())

    # Add ❌ Exit button
        exit_button = tk.Button(
        fs_window,
        text="❌ Exit Fullscreen",
        font=("Segoe UI", 10),
        bg="#333333",
        fg="white",
        command=fs_window.destroy,
        relief="flat"
    )
        exit_button.place(relx=0.98, rely=0.02, anchor="ne")  # top-right corner

    # Bring to front
        fs_window.lift()
        fs_window.focus_set()

    # Start the video player
        fs_player = CustomVideoPlayer(
        video_path=last_rendered_video_path,
        label_widget=fs_label,
        frame_delay_ms=100,
        loop=False,
        width=fs_window.winfo_screenwidth(),
        height=fs_window.winfo_screenheight()
    )
        fs_player.play()


# Overlay button on video_label
    fullscreen_btn = tk.Button(
    video_label,
    text="⛶ Fullscreen",
    command=toggle_fullscreen_video,
    font=("Segoe UI", 10),
    bg="#222222",
    fg="white",
    relief="flat"
)
    fullscreen_btn.place(relx=0.95, rely=0.02, anchor="ne")  # Top-right corner



    # Everything is now laid out: analyze_frame is packed, and its children are centered via grid(...)
    analyze_window.mainloop()

    
    
    
# GUI Setup
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("theme.json")    
    
# Create a new window for the analyze GUI
analyze_window = ctk.CTk()
analyze_window.title("🧠 Analyze EEG")
analyze_window.geometry(f"{analyze_window.winfo_screenwidth()}x{analyze_window.winfo_screenheight()-100}")
analyze_window.resizable(False, False)
analyze_frame = ctk.CTkScrollableFrame(
    analyze_window,
    width=analyze_window.winfo_screenwidth(),
    height=analyze_window.winfo_screenheight() - 100,
    fg_color="transparent"
)
result_label = ctk.CTkLabel(analyze_frame, text="", wraplength=500, font=("Consolas", 13), text_color="#000000")



root = ctk.CTk()
root.title("🎙️ Phoneme Pronouncer Pro")
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()-100}")
root.resizable(False, False)

# Fonts & Colors
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_NORMAL = ("Segoe UI", 12)
FONT_MONO = ("Courier New", 10)

# Title
title_label = ctk.CTkLabel(root, text="Phoneme Pronouncer Pro", font=FONT_TITLE, text_color="#380684")
title_label.pack(pady=20)

# Log Label
log_label = ctk.CTkLabel(root, text="📝 Conversation Log", font=("Segoe UI", 14, "bold"), text_color="#cba6f7")
log_label.pack(pady=(10, 0))

# Scrollable Text Widget for Log
log_frame = ctk.CTkFrame(root)
log_frame.pack(pady=5)

log_text = ctk.CTkTextbox(log_frame, height=200, width=600, font=("Consolas", 11))
log_text.pack(padx=10, pady=10)
log_text.configure(state='disabled')

# Entry Field
entry_label = ctk.CTkLabel(root, text="Type a word or phrase below:", font=FONT_NORMAL)
entry_label.pack()

entry = ctk.CTkEntry(root, font=("Segoe UI", 14), width=400, height=40)
entry.pack(pady=10)

# Buttons
global btn_pronounce
global btn_get_phonemes
global microgap_var
microgap_var = ctk.BooleanVar(value=True)  # default ON

btn_frame = ctk.CTkFrame(root)
btn_frame.pack(pady=10)

btn_get_phonemes = ctk.CTkButton(
    btn_frame,
    text="🔍 Get Phonemes",
    command=lambda: show_phonemes(analyze_window, analyze_frame, result_label),
    font=FONT_NORMAL,
    width=120,
    height=35,
    state="disabled"  # Start as disabled
)
btn_get_phonemes.pack(side="left", padx=10)



btn4 = ctk.CTkButton(btn_frame, text="🧠 Ask LaRocco", command=ask_larocco_gpt, font=FONT_NORMAL,
                     width=120, height=35)
btn4.pack(side="left", padx=10)

microgap_checkbox = ctk.CTkCheckBox(
    root,
    text="Insert Microgaps",
    variable=microgap_var,
    font=FONT_NORMAL,
    checkbox_height=20,
    checkbox_width=20,
    checkmark_color="white"  # Or any color that shows clearly
)

microgap_checkbox.pack(pady=(5, 0))








def convert_eeg_tsv_to_csv(input_tsv_path: str, output_csv_path: str):
    """
    Convert EEG .tsv file into .csv with original Index and cumulative Time in seconds.
    Each 256 samples = 1 second of EEG data.
    """
    headers = [
        "Index", "Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2", "F7", "F8", "C3", "C4",
        "T3", "T4", "P3", "P4", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2",
        "Other", "Other", "Other", "Other", "Other", "Other", "Other",
        "Analog Channel 0", "Analog Channel 1", "Analog Channel 2", "Timestamp", "Other"
    ]

    eeg_channels = [
        "Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
        "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"
    ]

    df = pd.read_table(input_tsv_path, sep="\t", header=None)
    df.columns = headers

    num_rows = len(df)

    # Time in seconds: 256 rows = 1 second
    time_col = [round(i / 256, 8) for i in range(num_rows)]

    # Extract EEG + add Time, keep original Index from TSV
    df_eeg = df[["Index"] + eeg_channels].copy()
    df_eeg.insert(1, "Timestamp", time_col)

    df_eeg.to_csv(output_csv_path, index=False)


# Run Application
root.mainloop()
