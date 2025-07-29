# 🚀 Prodigal Automation

A robust **Retrieval-Augmented Generation (RAG)** system that **automatically fills out startup application forms** using multi-format document input (PDF, PPTX, TXT, images) and AI-generated answers — with an optional manual review workflow.

## 💡 What It Does

1. You provide your **startup/company documents** (pitch decks, business plans, screenshots, notes).
2. The system reads and processes the content using OCR, NLP, and vector search.
3. It **automatically generates answers** to application or funding questions.
4. You can optionally **review, edit, and approve** answers interactively.
5. The final output is ready to be used in accelerator applications like **Y Combinator**, **Techstars**, or grant/funding forms.

---

## 🧠 Key Features

- ✅ **Multi-Format Document Ingestion**: Supports PDF, TXT, PPTX, and image files (PNG, JPG, JPEG, TIFF, BMP)
- 🔍 **Advanced OCR**: Dual OCR engine (EasyOCR + Tesseract) for high-accuracy text extraction
- 📑 **PDF + Embedded Image Text**: Extracts text from both standard and image-based PDFs
- 📊 **PowerPoint Processing**: Parses text and tables from slides
- 💬 **AI-Powered Question Answering**: Uses GPT-4 to generate answers from document context
- 📝 **Interactive Review Flow**: Accept, edit, or skip AI-generated answers manually
- 📈 **Confidence Scoring**: Scores AI responses for prioritization
- 💾 **JSON-based Persistence**: Load/save Q&A states for future reuse

---

## 🖥️ System Requirements

- Python 3.8+
- Tesseract OCR installed
- OpenAI API key

#### Tesseract Installation
**macOS**
```bash
brew install tesseract
```
**Ubuntu/Debian**
```bash
sudo apt-get install tesseract-ocr
```
**Windows**  
Install from: https://github.com/UB-Mannheim/tesseract/wiki

---

## ⚙️ Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/prodigal-automation.git
   cd prodigal-automation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   - Create a `.env` file:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. Configure Tesseract path if needed (line 33 in `main.py` or similar).

---

## 🗂️ File Structure

```
prodigal-automation/
├── main.py
├── .env
├── questions.json
├── generated_qa_pairs.json
├── reviewed_qa_pairs.json
└── requirements.txt
```

---

## ✍️ Startup Workflow (Flowchart Description)

> **Input**: Company provides documents (pitch deck, business plan, team bios, etc.)  
> **Step 1**: System extracts relevant questions from application forms  
> **Step 2**: Documents are processed using OCR and NLP  
> **Step 3**: RAG pipeline retrieves context & generates AI answers  
> **Step 4** *(Optional)*: User reviews and edits responses  
> **Output**: JSON file with finalized responses, ready for form submission

---

## 📥 How to Use

### 1. Prepare Questions
```json
[
  "What problem are you solving?",
  "Who writes the code?",
  "What is your business model?",
  "How do you plan to scale?"
]
```

### 2. Collect Documents
Gather all your startup-related documents: PDFs, PPTX, images, etc.

### 3. Run the Tool
```bash
python main.py
```

### 4. Use the Menu
```
1. Process documents & generate answers
2. Review answers
3. Exit
```

---

## 🔄 Review Commands

| Key | Action |
|-----|--------|
| `a` | Accept answer |
| `e` | Edit answer manually |
| `s` | Skip (keep AI answer) |
| `q` | Quit review |
| `h` | Help menu |

---

## 📦 Output

### `generated_qa_pairs.json`
Raw AI-generated answers with confidence scores.

### `reviewed_qa_pairs.json`
Finalized answers after review. Example:
```json
[
  {
    "question": "How will you make money?",
    "ai_answer": "Based on the pitch deck, the company plans to monetize through...",
    "manual_answer": "We offer subscription-based pricing for enterprise clients.",
    "has_manual_input": true,
    "confidence_score": 0.85,
    "edited": true
  }
]
```

---

## 📚 Use Cases

- **Accelerator Applications** (e.g., YC, Techstars)
- **Startup Grant Forms**
- **VC Due Diligence Preparation**
- **Investor Q&A Document Automation**
- **FAQ Generation from Technical Docs**

---

## ⚙️ Configuration Tips

```python
chunk_size=1000
chunk_overlap=200
model="gpt-4o"
temperature=0.1
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---

## 🧰 Troubleshooting

| Issue | Solution |
|-------|----------|
| Tesseract not found | Verify installation & update `tesseract_cmd` |
| OpenAI API key error | Check `.env` and account credits |
| OCR misses text | Improve image quality or switch engines |
| High memory usage | Lower `chunk_size` or reduce document count |

---

## 🔐 Security

- Never commit your `.env` file
- Do not expose confidential startup documents
- Review OpenAI’s data usage policies

---

## 📝 License

Open source under [your chosen license]. Check OpenAI and dependency licenses for commercial use.

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 📩 Need Help?

- Check the Troubleshooting section
- Read inline code comments
- Open an issue with logs and error details

---

**Build smart, submit smarter. 🚀 Prodigal Automation is your startup co-pilot!**