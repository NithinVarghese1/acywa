import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import traceback
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction
app.secret_key = 'your_secret_key_for_session_management'

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")

# Assistant class with API key integration
class Assistant:
    def __init__(self, file_path, context):
        self.context = context
        self.docs = self.load_text(file_path)
        self.vectorStore = self.create_db(self.docs)
        self.chain = self.create_chain()
        self.chat_history = []
        self.is_new_user = False

    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return Chroma.from_documents(docs, embedding=embedding)

    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=openai_api_key
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant chatbot specifically focused on giving a tutorial on how to navigate the Atlas map, based on {context}. Your primary goal is to help users with {context} only."),
            ("system", "Context: {context}"),
            ("system", "Instructions for {context}:"
                       "\n1. If given a one-word or vague query, ask for clarification before proceeding."
                       "\n2. For all users, provide the following general steps for finding data on a specific theme or indicator:"
                       "\n   - Direct users to open the Atlas maps"
                       "\n   - Instruct users to use the theme or indicator search box in Atlas maps"
                       "\n   - Explain that if data is available on the topic, it will appear as a dropdown"
                       "\n   - Do not interpret specific data or findings"
                       "\n3. Always relate your responses back to the user's original query, regardless of the theme or indicator."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", "Remember to be concise, clear, and helpful in your responses - give a maximum of 3 sentences. "
                       "Focus exclusively on {context} and do not discuss other topics unless explicitly asked."
                       "After giving guidance, suggest two relevant follow-up questions.")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt,
            document_variable_name="context"  # This ensures context is passed correctly as an input
        )

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Given the above conversation about {context}, generate a search query to look up relevant information")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        return create_retrieval_chain(
            history_aware_retriever,
            chain
        )

    def process_chat(self, question):
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context  # Pass context explicitly here
        })
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    def reset_chat_history(self):
        self.chat_history = []

class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('Raw data - maps.txt', 'map navigation')


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    # Check if session state exists, otherwise initialize
    if 'chat_state' not in session:
        session['chat_state'] = {'is_new_user': True, 'step': 1}

    chat_state = session['chat_state']

    try:
        # Interactive logic similar to CLI mode in your original code
        if chat_state['step'] == 1:
            bot_reply = "Hello! Welcome to the Atlas Map Navigation Assistant! Are you new to our interactive map platform? (Yes/No)"
            chat_state['step'] = 2
        elif chat_state['step'] == 2:
            if user_message.lower() in ['yes', 'y']:
                chat_state['is_new_user'] = True
                bot_reply = "Great! Let's start by familiarizing you with the map platform. Follow these steps: 1. Click on Atlas maps 2. Navigate to the right-hand side pane 3. Click the 'i' icon in the top right-hand corner."
                chat_state['step'] = 3
            else:
                chat_state['is_new_user'] = False
                bot_reply = "Welcome back! What specific question can I assist you with first?"
                chat_state['step'] = 3
        elif chat_state['step'] == 3:
            bot_reply = "What specific question can I assist you with today?"
            # After this, you can continue handling specific questions in further steps.

        # Update session with the current state
        session['chat_state'] = chat_state

        return jsonify({"reply": bot_reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": "Sorry, there was an error processing your request."}), 500

# Main entry point for Flask API
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
