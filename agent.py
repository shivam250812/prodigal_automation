from browser_use.llm import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import os
import json
from playwright.async_api import async_playwright

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

def determine_yes_no_from_answer(answer_text):
    """Determine if an answer indicates Yes or No based on the text content."""
    if not answer_text:
        return False
    
    answer_lower = answer_text.lower()
    
    # Check for explicit "No" at the beginning or as a standalone word
    if answer_lower.startswith('no') or ' no ' in answer_lower or answer_lower.endswith(' no'):
        return False
    
    # Check for explicit "Yes" at the beginning or as a standalone word
    if answer_lower.startswith('yes') or ' yes ' in answer_lower or answer_lower.endswith(' yes'):
        return True
    
    # Keywords that indicate "Yes" (lower priority)
    yes_keywords = [
        'have', 'has', 'do', 'does', 'currently', 'active', 
        'revenue', 'users', 'customers', 'incorporated', 'investment', 
        'funding', 'raising', 'fundraising', 'investors', 'angel', 'venture',
        'seed', 'series', 'funded', 'received', 'got', 'obtained'
    ]
    
    # Keywords that indicate "No" (lower priority)
    no_keywords = [
        'false', 'not', "don't", "doesn't", 'none', 'zero', '0',
        'not yet', 'planning', 'future', 'will', 'going to', 'intend to'
    ]
    
    # Check for other No keywords
    for keyword in no_keywords:
        if keyword in answer_lower:
            return False
    
    # Check for Yes keywords (lower priority)
    for keyword in yes_keywords:
        if keyword in answer_lower:
            return True
    
    # Default to False if unclear
    return False

def extract_radio_button_answers(qa_pairs):
    """Extract Yes/No answers for radio button questions from Q&A pairs."""
    radio_button_mapping = {
        "Are people using your product?": "stage",
        "Do you have revenue?": "revenue", 
        "Have you formed ANY legal entity yet?": "incorporation",
        "Have you taken any investment yet?": "investment",
        "Are you currently fundraising?": "raising"
    }
    
    extracted_answers = {}
    
    for qa_pair in qa_pairs:
        question = qa_pair.get('question', '').strip()
        
        if question in radio_button_mapping:
            # Get the answer (manual or AI)
            manual_answer = qa_pair.get('manual_answer')
            ai_answer = qa_pair.get('ai_answer', '')
            
            answer_text = manual_answer if manual_answer and manual_answer.strip() else ai_answer
            
            # Determine Yes/No from the answer
            is_yes = determine_yes_no_from_answer(answer_text)
            
            extracted_answers[question] = {
                'type': radio_button_mapping[question],
                'answer_text': answer_text,
                'is_yes': is_yes,
                'qa_pair': qa_pair
            }
            
            print(f"🎯 Extracted radio button answer: {question}")
            print(f"📝 Answer: {answer_text[:100]}...")
            print(f"✅ Determined: {'YES' if is_yes else 'NO'}")
            print("---")
    
    return extracted_answers

async def click_radio_button(page, question_type, is_yes):
    """Click radio button with the working method from debug_form_filler.py."""
    radio_selectors = {
        "stage": {
            "yes": 'label[for="stage-2"]',
            "no": 'label[for="stage-1"]'
        },
        "revenue": {
            "yes": 'label[for="revenue-yes"]',
            "no": 'label[for="revenue-no"]'
        },
        "incorporation": {
            "yes": 'label[for="incyet-yes"]',
            "no": 'label[for="incyet-no"]'
        },
        "investment": {
            "yes": 'label[for="investyet-yes"]',
            "no": 'label[for="investyet-no"]'
        },
        "raising": {
            "yes": 'label[for="currentlyraising-yes"]',
            "no": 'label[for="currentlyraising-no"]'
        }
    }
    
    if question_type not in radio_selectors:
        print(f"❌ Unknown question type: {question_type}")
        return False
    
    yes_no = "yes" if is_yes else "no"
    selector = radio_selectors[question_type][yes_no]
    
    try:
        await page.locator(selector).click()
        print(f"✅ Clicked radio button for {question_type}: {yes_no.upper()}")
        return True
    except Exception as e:
        print(f"❌ Failed to click radio button for {question_type}: {e}")
        return False

async def setup_form_with_radio_buttons():
    """Set up the form and click radio buttons using Playwright, then return the page URL."""
    print("🎯 Setting up form and clicking radio buttons with Playwright...")
    
    # Load Q&A pairs
    qa_pairs = load_qa_pairs()
    if not qa_pairs:
        print("❌ No Q&A pairs found.")
        return None
    
    # Extract radio button answers
    radio_answers = extract_radio_button_answers(qa_pairs)
    print(f"🎯 Extracted {len(radio_answers)} radio button answers")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page()
        
        try:
            # Navigate to YC application
            print("🌐 Navigating to YC application...")
            await page.goto("https://apply.ycombinator.com/")
            
            # Click Apply Now button
            print("📝 Clicking Apply Now...")
            await page.click("a.btn.apply")
            
            # Login
            print("🔐 Logging in...")
            await page.fill("#ycid-input", yc_email)
            await page.fill("#password-input", yc_password)
            await page.click("button[type='submit']")
            
            # Wait for login and click Finish application
            await page.wait_for_load_state("networkidle")
            print("📋 Clicking Finish application...")
            await page.click("text=Finish application")
            await page.wait_for_load_state("networkidle")
            
            # Process radio buttons
            print("🎯 Processing radio buttons...")
            for question, info in radio_answers.items():
                question_type = info['type']
                is_yes = info['is_yes']
                
                print(f"🎯 Processing: {question}")
                print(f"✅ Will click: {'YES' if is_yes else 'NO'}")
                
                success = await click_radio_button(page, question_type, is_yes)
                
                if success:
                    print(f"✅ Successfully processed: {question}")
                else:
                    print(f"❌ Failed to process: {question}")
                
                await page.wait_for_timeout(1000)
            
            print("✅ Radio button processing completed!")
            
            # Get the current URL and page state
            current_url = page.url
            print(f"🔗 Current URL: {current_url}")
            
            # Keep the browser open and return the URL
            print("🔍 Browser will remain open for browser_use to continue...")
            return current_url
            
        except Exception as e:
            print(f"❌ Error during setup: {e}")
            return None

# Load Q&A pairs
qa_pairs = load_qa_pairs()
if not qa_pairs:
    print("❌ No Q&A pairs found. Please run the RAG system first.")
    exit(1)

# Create answer mapping
answer_mapping = create_answer_mapping(qa_pairs)
print(f"✅ Loaded {len(answer_mapping)} answers from reviewed_qa_pairs.json")

# Extract radio button answers
radio_answers = extract_radio_button_answers(qa_pairs)
print(f"🎯 Extracted {len(radio_answers)} radio button answers from Q&A pairs")

# Convert answers to JSON string for the task
answers_json = json.dumps(answer_mapping, indent=2)

# Create radio button summary
radio_summary = ""
for question, info in radio_answers.items():
    radio_summary += f"- {question}: {'YES' if info['is_yes'] else 'NO'} ({info['type']})\n"

task = f"""
You are a browser automation agent tasked with filling the Y Combinator application form using provided answers. Follow these instructions carefully:

1. Navigate to the Y Combinator application form URL: https://apply.ycombinator.com/app/edit
2. You should see that the radio buttons are already clicked on this page
3. Fill the form fields using the provided answers

ANSWERS TO USE:
{answers_json}

RADIO BUTTON ANSWERS EXTRACTED FROM Q&A PAIRS:
{radio_summary}

IMPORTANT: The radio buttons have already been clicked automatically using Playwright. 
You should see that the following radio buttons are already selected:
- Are people using your product?: YES
- Do you have revenue?: NO  
- Have you formed ANY legal entity yet?: YES
- Have you taken any investment yet?: NO
- Are you currently fundraising?: NO

FILLING INSTRUCTIONS:
- Navigate to the application form URL
- The radio buttons should already be clicked on the page
- Focus on filling the remaining form fields on the current page
- For each form field, find the matching question in the answers above
- If a manual answer exists (not null/empty), use the manual answer
- If manual answer is null/empty, use the AI answer
- For text areas and inputs, paste the full answer text
- Be careful with character limits - truncate if necessary
- For questions not in the answers, leave blank or use "Not applicable"
- Fill all sections of the application form
- Do NOT submit the form - just fill it with the answers
- Scroll down to access all form sections as needed

FILL THE FORM NOW using the answers provided above.
"""

async def main():
    # First, set up the form and click radio buttons
    print("🚀 Starting YC Application Form Filler")
    print("=" * 60)
    
    form_url = await setup_form_with_radio_buttons()
    
    if form_url:
        print("✅ Form setup completed successfully!")
        print("🤖 Now starting browser_use agent for form filling...")
        print(f"🔗 Form URL: {form_url}")
        
        # Then run the browser_use agent for form filling
        agent = Agent(task=task, llm=llm)
        result = await agent.run()
        print("Form filling completed!")
    else:
        print("❌ Form setup failed. Stopping.")

if __name__ == "__main__":
    asyncio.run(main())

