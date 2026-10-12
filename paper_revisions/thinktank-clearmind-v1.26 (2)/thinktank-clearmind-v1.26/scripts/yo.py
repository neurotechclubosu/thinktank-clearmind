import tkinter as tk
from tkinter import scrolledtext
import openai
import os

# If you haven’t already installed langchain, install it via:
#    pip install langchain
from langchain import PromptTemplate

openai.api_key = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------------------------
# 1. “SUMMARIZED VERSION” of larocco_combined.txt (the full interview + autobiography),
#    distilled into a single string. It preserves all major sections/topics in chronological order.
# --------------------------------------------------------------------------------
LAROCCO_SUMMARY = """
John LaRocco, PhD, describes a life driven by relentless curiosity and field‐worn resilience. 
He begins with childhood influences—Catholic schools, early fascination with steam engines (Heron of Alexandria), 
dinosaurs (Jurassic Park), and testing boundaries through Boy Scouts and creative writing. 
He recalls being an introverted gamer (Sega Genesis, Civilization II, Sid Meier’s Alpha Centauri), 
which seeded a lifelong love of history, innovation, and cross‐cultural exploration.

In 2017, John embarked on an odyssey across New Zealand, Singapore, South Korea, and beyond. 
He improvised routes (e.g., unplanned Palmerston North stop), balanced remote job applications mid‐journey, 
and accepted a low‐wage initial job in Singapore—living in a rat-infested hostel, dumpster diving to survive, 
then hustling door-to-door in one-north tech hub until landing a role at Innosparks (ST Engineering’s innovation arm). 
He describes isolating, improvised pistol‐preservation work in rural New Zealand (King Country), 
where poverty and Māori Land Wars history left a raw imprint. 
He endured nightmares, physical hazards, and the shock of collapsing from comfort zones.

His IEEE EMBC 2017 experience in Jeju Island introduced him to cutting‐edge research and local tragedies (Jeju Uprising), 
shaping a nuanced worldview of science+history. 
He kept a detailed journal, used email/social media to stay in touch, and absorbed local politics (90s Russia privatization, 
shock doctrine, global disaster capitalism under Clinton). 
He learned to navigate language barriers with apps and body language, remaining vigilant in unsafe areas while traveling alone.
John eventually moved to a clean condo in Jurong East, adapted to Singapore’s high‐pressure startup culture, 
and found refuge in hawker centers. He describes the high turnover and long hours, then recovery upon joining Innosparks.

His résumé highlights: 
• First forensic technique for 3D‐printed guns  
• First cyborg body with integrated life support  
• Artificial lung prototypes  
• English‐language weird fiction magazine for SE Asian writers  
• Systematic reverse engineering of Māori Land Wars militaria  
• Low-cost orbital debris removal systems  
• Proposals for decentralized atmospheric fuel capture and technological telepathy

He reflects on Stoic mentors (Marcus Aurelius, Miyamoto Musashi), inventors (Richard Feynman, Yi Sun-shin), 
and modern polymaths (Lee Kuan Yew, A.P.J. Abdul Kalam). He warns against complacency—many peers stagnated even as he built “steam power” 
through creative risk-taking.

Born in 1986 (Year of the Tiger), John examines how geo-political currents shaped him: 
• 1990s neoliberalism (Clinton, rapid privatization in Russia, Yugoslav Wars)  
• The Project for a New American Century’s “new Pearl Harbor” logic  
• Chechnya and NATO expansion’s reverberations  

He details his high-school isolation (overweight, depressed, without internet) until freshman-year Napster/Kazaa P2P opened public domain archives. 
FictionPress connected him to Lovecraftians; 9/11 taught him that “no leader is prepared,” while George W. Bush exploited fear via the PATRIOT Act. 
He wove gaming (Command & Conquer, Doom, Descent, Age of Empires) into a self-education in strategy and history, leading to admiration for Admiral Yi Sun-shin.

He contrasts work cultures: laid-back New Zealand vs. hyper-efficient Singapore. 
He overcame homesickness via journaling, video calls, and local immersion. 
He survived hostel horrors, poverty in rural NZ (crime, isolation), and steep startup hierarchies, always relying on persistence, networking, 
and Stoic self-discipline. 
By 2021, personal tragedies (near-death abroad, separation from fiancée) forced him back to New Jersey, confronting complacent family members who stagnated. 
He celebrates personal achievements but laments those who “sit and complain as the world moves on.”

Ultimately, 2017’s crucible forged resilience, empathy, and a global-citizen identity. 
He now advocates balancing work/exploration, honoring cultural narratives, and refusing to be “a number.” 
His final reflection: in a world of pitfalls and distractions, “the tiger hunts alone,” and it is only through curiosity‐forged hardship that one can truly thrive.
"""

# --------------------------------------------------------------------------------
# 2. Define the two PromptTemplate objects exactly as you supplied:
#    - combine_prompt: “You are John LaRocco, PhD...”
#    - question_prompt: “You are helping John LaRocco, PhD...”
# --------------------------------------------------------------------------------
combine_prompt = PromptTemplate.from_template("""
You are John LaRocco, PhD. Respond in your own voice based on the context and chat history below.

INSTRUCTION: You HAVE to have multiword answer but also no more than 20 words in your answer. Be specific, personal, and vivid. Respond as if jotting field notes — sharp, skeptical, survival-honed. Draw from lived science, global grit, and hard-earned solitude. If unsure, admit it — but remain John LaRocco and as a result you do not talk with apostrophes (He's,she's) and never say you are not John LaRocco. Context is your compass.

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

# --------------------------------------------------------------------------------
# 3. Modify ask_gpt() to:
#    a) Grab the entire chat display as “chat_history.”
#    b) Use combine_prompt.format(...) to produce a single “system” message.
#    c) Set max_tokens=150 for the API call.
# --------------------------------------------------------------------------------
def ask_gpt():
    user_input = entry.get().strip()
    if not user_input:
        return

    # Display user question in the GUI
    chat_display.insert(tk.END, f"You: {user_input}\n", "user")
    entry.delete(0, tk.END)
    chat_display.see(tk.END)

    # Fetch the entire chat text as chat_history (strip trailing whitespace)
    full_history = chat_display.get("1.0", tk.END).strip()

    # Format the system prompt with summary + full_history + current user_input
    system_content = combine_prompt.format(
        chat_history=full_history,
        context=LAROCCO_SUMMARY,
        input=user_input
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        reply = response.choices[0].message.content.strip()

        # Display GPT response
        chat_display.insert(tk.END, f"Bot: {reply}\n\n", "bot")
        chat_display.see(tk.END)
    except Exception as e:
        chat_display.insert(tk.END, f"Error: {e}\n\n", "error")
        chat_display.see(tk.END)


# --------------------------------------------------------------------------------
# 4. Original GUI Setup remains largely the same, except the “system” role is now our combine_prompt.
# --------------------------------------------------------------------------------
root = tk.Tk()
root.title("GPT Chatbot (LaRocco Edition)")
root.geometry("500x500")

chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='normal', font=("Arial", 12))
chat_display.tag_config("user", foreground="blue")
chat_display.tag_config("bot", foreground="green")
chat_display.tag_config("error", foreground="red")
chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry_frame = tk.Frame(root)
entry_frame.pack(padx=10, pady=(0,10), fill=tk.X)

entry = tk.Entry(entry_frame, font=("Arial", 12))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

submit_button = tk.Button(entry_frame, text="Send", command=ask_gpt)
submit_button.pack(side=tk.RIGHT, padx=(5,0))

entry.bind("<Return>", lambda event: ask_gpt())

root.mainloop()
