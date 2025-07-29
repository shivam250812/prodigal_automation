from browser_use.llm import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import os
import json

load_dotenv()
yc_email = os.getenv("YC_USERNAME")
yc_password = os.getenv("YC_PASSWORD")

llm = ChatOpenAI(model="gpt-4.1")

def load_qa_pairs(filename="reviewed_qa_pairs.json"):
    """Load Q&A pairs from JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"❌ File not found: {filename}")
            return []
    except Exception as e:
        print(f"❌ Error loading Q&A pairs: {e}")
        return []

def create_answer_mapping(qa_pairs):
    """Create a mapping of questions to answers."""
    answer_map = {}
    
    for qa_pair in qa_pairs:
        question = qa_pair.get('question', '').strip()
        manual_answer = qa_pair.get('manual_answer')
        ai_answer = qa_pair.get('ai_answer', '')
        
        # Use manual answer if available, otherwise use AI answer
        if manual_answer and manual_answer.strip():
            answer_map[question] = manual_answer.strip()
        else:
            answer_map[question] = ai_answer.strip()
    
    return answer_map

# Load Q&A pairs
qa_pairs = load_qa_pairs()
if not qa_pairs:
    print("❌ No Q&A pairs found. Please run the RAG system first.")
    exit(1)

# Create answer mapping
answer_mapping = create_answer_mapping(qa_pairs)
print(f"✅ Loaded {len(answer_mapping)} answers from reviewed_qa_pairs.json")

# Convert answers to JSON string for the task
answers_json = json.dumps(answer_mapping, indent=2)

task = f"""
You are a browser automation agent tasked with filling the Y Combinator application form using provided answers. Follow these instructions carefully:

1. Visit https://www.ycombinator.com/apply/
2. Click "Apply Now".
3. Log in using:
   - Email: {yc_email}
   - Password: {yc_password}
4. Click "Finish Application".
5. Navigate through the application form and fill it using the provided answers.

ANSWERS TO USE:
{answers_json}

FILLING INSTRUCTIONS:
- For each form field, find the matching question in the answers above
- If a manual answer exists (not null/empty), use the manual answer
- If manual answer is null/empty, use the AI answer
- For radio buttons, match the answer text to the option
- For text areas and inputs, paste the full answer text
- Be careful with character limits - truncate if necessary
- For questions not in the answers, leave blank or use "Not applicable"
- Fill all sections of the application form
- Do NOT submit the form - just fill it with the answers

FILL THE FORM NOW using the answers provided above.
"""

async def main():
    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    print("✅ Form filling completed!")

if __name__ == "__main__":
    asyncio.run(main()) 