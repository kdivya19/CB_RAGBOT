from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from final_helper import get_qa_chain

app = FastAPI(title="Codebasics Q&A API")

# 1. Load Model at Startup (Don't load inside the function, it's slow)
# Global variable to hold the chain
chain = get_qa_chain()

# 2. Define Input Structure (Data Validation)
class QuestionRequest(BaseModel):
    text: str

# 3. API Endpoint (for developers / other apps)
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Calling the chain
        response = chain.invoke(request.text)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Health Check
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# 5. Frontend — HTML Chat Interface
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codebasics Q&A — AI Assistant</title>
    <meta name="description" content="Ask questions about Codebasics courses and get instant AI-powered answers.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 720px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.4);
        }

        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .header .logo {
            font-size: 40px;
            margin-bottom: 8px;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: rgba(255, 255, 255, 0.5);
            font-size: 14px;
            font-weight: 300;
        }

        .input-group {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }

        #question-input {
            flex: 1;
            padding: 16px 20px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 14px;
            color: #fff;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            outline: none;
            transition: all 0.3s ease;
        }

        #question-input:focus {
            border-color: #a78bfa;
            background: rgba(255, 255, 255, 0.12);
            box-shadow: 0 0 20px rgba(167, 139, 250, 0.15);
        }

        #question-input::placeholder {
            color: rgba(255, 255, 255, 0.3);
        }

        #ask-btn {
            padding: 16px 28px;
            background: linear-gradient(135deg, #a78bfa, #7c3aed);
            border: none;
            border-radius: 14px;
            color: #fff;
            font-size: 15px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        #ask-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
        }

        #ask-btn:active {
            transform: translateY(0);
        }

        #ask-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .chat-area {
            max-height: 400px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding-right: 8px;
        }

        .chat-area::-webkit-scrollbar {
            width: 4px;
        }

        .chat-area::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }

        .message {
            padding: 16px 20px;
            border-radius: 16px;
            animation: fadeIn 0.4s ease;
            line-height: 1.6;
            font-size: 14px;
        }

        .message.question {
            background: rgba(167, 139, 250, 0.15);
            border: 1px solid rgba(167, 139, 250, 0.25);
            color: #c4b5fd;
            align-self: flex-end;
            max-width: 85%;
        }

        .message.question::before {
            content: "You: ";
            font-weight: 600;
            color: #a78bfa;
        }

        .message.answer {
            background: rgba(52, 211, 153, 0.1);
            border: 1px solid rgba(52, 211, 153, 0.2);
            color: #a7f3d0;
            align-self: flex-start;
            max-width: 85%;
        }

        .message.answer::before {
            content: "AI: ";
            font-weight: 600;
            color: #34d399;
        }

        .message.error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #fca5a5;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 16px 20px;
            color: rgba(255, 255, 255, 0.5);
            font-size: 14px;
        }

        .loading .dots span {
            animation: blink 1.4s infinite;
            font-size: 20px;
        }

        .loading .dots span:nth-child(2) { animation-delay: 0.2s; }
        .loading .dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes blink {
            0%, 80%, 100% { opacity: 0.2; }
            40% { opacity: 1; }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: rgba(255, 255, 255, 0.25);
            font-size: 14px;
        }

        .empty-state .icon {
            font-size: 48px;
            margin-bottom: 12px;
        }

        .footer {
            text-align: center;
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            color: rgba(255, 255, 255, 0.25);
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🎓</div>
            <h1>Codebasics Q&A</h1>
            <p>AI-powered answers about Codebasics courses</p>
        </div>

        <div class="input-group">
            <input
                type="text"
                id="question-input"
                placeholder="Ask anything about Codebasics..."
                autocomplete="off"
            >
            <button id="ask-btn" onclick="askQuestion()">Ask ✨</button>
        </div>

        <div class="chat-area" id="chat-area">
            <div class="empty-state" id="empty-state">
                <div class="icon">💬</div>
                <p>Ask a question to get started!</p>
            </div>
        </div>

        <div class="footer">
            Powered by LangChain + Gemini | Codebasics FAQ Assistant
        </div>
    </div>

    <script>
        const input = document.getElementById('question-input');
        const chatArea = document.getElementById('chat-area');
        const askBtn = document.getElementById('ask-btn');
        const emptyState = document.getElementById('empty-state');

        // Allow Enter key to submit
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !askBtn.disabled) {
                askQuestion();
            }
        });

        async function askQuestion() {
            const question = input.value.trim();
            if (!question) return;

            // Remove empty state
            if (emptyState) emptyState.remove();

            // Show user's question
            addMessage(question, 'question');
            input.value = '';

            // Show loading
            const loadingEl = showLoading();

            // Disable button
            askBtn.disabled = true;
            askBtn.textContent = 'Thinking...';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: question })
                });

                const data = await response.json();

                loadingEl.remove();

                if (response.ok) {
                    addMessage(data.answer, 'answer');
                } else {
                    addMessage('Error: ' + (data.detail || 'Something went wrong'), 'error');
                }
            } catch (err) {
                loadingEl.remove();
                addMessage('Connection error. Please try again.', 'error');
            }

            askBtn.disabled = false;
            askBtn.textContent = 'Ask ✨';
            input.focus();
        }

        function addMessage(text, type) {
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.textContent = text;
            chatArea.appendChild(div);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function showLoading() {
            const div = document.createElement('div');
            div.className = 'loading';
            div.innerHTML = 'Thinking <span class="dots"><span>.</span><span>.</span><span>.</span></span>';
            chatArea.appendChild(div);
            chatArea.scrollTop = chatArea.scrollHeight;
            return div;
        }
    </script>
</body>
</html>
"""