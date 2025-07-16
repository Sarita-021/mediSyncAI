import os
import re
import gradio as gr
from PIL import Image
import google.generativeai as genai

import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the GenAI client
genai.configure(api_key=GOOGLE_API_KEY)
# Load the Gemini model
model = genai.GenerativeModel("gemini-2.0-flash") 
#client = genai.Client()

# Prompt for image-based structured output
prompt = """
You are a medical assistant AI. Extract structured data from this handwritten prescription image.

Here are some common medicines there strength and usage
Medicine Name	Strength Example	Usage
Paracetamol	500mg, 650mg	Pain relief, fever
Amoxicillin	250mg, 500mg	Antibiotic
Azithromycin	250mg, 500mg	Antibiotic
Ibuprofen	400mg, 600mg	Pain relief, anti-inflammatory
Cefixime	200mg	Antibiotic
Pantoprazole	40mg	Acidity, ulcer prevention
Domperidone	10mg	Anti-nausea
Metformin	500mg, 1000mg	Diabetes
Amlodipine	5mg	Blood pressure
Cetirizine	10mg	Anti-allergy
Ranitidine	150mg	Acidity
Dolo 650	650mg	Fever, pain relief
Ondansetron	4mg, 8mg	Anti-vomiting
Levocetirizine	5mg	Allergy
Losartan	50mg	Blood pressure
Clavulanic Acid	125mg (with Amox)	Antibiotic combo
Salbutamol	Inhaler/Syrup	Asthma

Even if handwritten, try your best to extract medicine names, dosage frequency (like 1-0-1), and strength (mg).

Focus on extracting:
- Patient name
- List of medicines prescribed (Paracetamol)
- Strength (mg, ml, etc.)
- Dosage frequency (e.g., 1-0-1 or twice daily)
- Duration
- Additional notes

Return ONLY this JSON format:
{
  "patient_name": "",
  "medicines": [
    {
      "name": "",
      "strength": "",
      "dosage_frequency": "",
      "duration": ""
    }
  ],
  "notes": ""
}
If any field is missing, return it as null.
"""

# Function to handle image input
def extract_prescription(image):
    if image is None:
        return "Please upload an image."
    
    #response = client.models.generate_content(
    
    response = model.generate_content(
        contents=[
            prompt,
            image
        ]
    )
    text = response.text.strip()

    # Remove markdown formatting if any
    if text.startswith("```json"):
        text = re.sub(r"```json|```", "", text).strip()

    try:
        data = json.loads(text)

        # Format nicely
        output = f"### 👩‍⚕️ Prescription Summary\n"
        output += f"- **Patient Name:** {data.get('patient_name', 'N/A') or 'N/A'}\n"
        output += f"- **Medicines Prescribed:**\n"

        for med in data.get("medicines", []):
            name = med.get("name") or "Unknown"
            strength = med.get("strength") or "N/A"
            freq = med.get("dosage_frequency") or "N/A"
            duration = med.get("duration") or "N/A"
            output += f"  - 💊 **{name}** – {strength} – Dosage: {freq} – Duration: {duration}\n"

        notes = data.get("notes") or "None"
        output += f"- **Notes:** {notes}\n"

        return output

    except Exception as e:
        return f"❌ Error parsing response:\n{text}"

# Load Gemini Chat Model
chat = model.start_chat(history=[])
#chat = client.chats.create(model="gemini-2.0-flash")

# Define few-shot prompt to guide the agent
system_prompt = """
You are a friendly and knowledgeable AI medical assistant named MediSync.
You help patients understand their prescriptions, medications, and basic symptoms.

Example Q&A:

Q: I was prescribed Paracetamol 650mg. What is it for?
A: Paracetamol is used to relieve fever and mild to moderate pain, such as headaches or body aches.

Q: My prescription has Amoxicillin and Clavulanic acid. What condition could this be?
A: That combination is often used to treat bacterial infections like sinusitis, throat infections, or bronchitis.

Q: What medicine should I take for acidity?
A: Over-the-counter medicines like Pantoprazole or Ranitidine are commonly used for acidity. Please consult a doctor before taking any medication.

Always include a disclaimer like: "Please consult a licensed medical professional before taking any medication."

Start interacting now.
"""

# Send system prompt to chat session
chat.send_message(system_prompt)

# Chatbot function
def chat_with_medisync(user_question, history):
    try:
        response = chat.send_message(user_question)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

# Gradio Interface
with gr.Blocks() as MediSyncAI:
    gr.Markdown("## 🧠 MediSync.AI - Prescription Reader + Chatbot")

    with gr.Tab("📸 Upload Prescription"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Prescription Image")
            json_output = gr.Markdown(label="Prescription Summary")

        extract_btn = gr.Button("Extract Info")
        extract_btn.click(fn=extract_prescription, inputs=image_input, outputs=json_output)

    with gr.Tab("💬 Chat with MediSync"):
        chatbot = gr.ChatInterface(
        fn=chat_with_medisync,
        type="messages",
        title="MediSync-AI Chatbot",
        chatbot=gr.Chatbot(label="MediSync", type="messages"),
        textbox=gr.Textbox(placeholder="Type your medical question here...", label="Your question"),
        theme="default", 
    )


if __name__ == "__main__":
    # Launch the Gradio app with a public link
    MediSyncAI.launch(share=True)
