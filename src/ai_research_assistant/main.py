import streamlit as st
from utils import extract_text_from_pdf, index_document
from agent import create_agent, invoke

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent, st.session_state.config = create_agent()
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

def create_sidebar():
    with st.sidebar:
        st.title("📚 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=False,
            help="Upload PDF documents to chat with them"
        )

        if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner("Processing document..."):
                try:
                    # Extract text from PDF
                    text = extract_text_from_pdf(uploaded_file)

                    # Index document in Pinecone
                    # doc_id = str(uuid.uuid4())
                    index_document(text)

                    st.session_state.uploaded_files.add(uploaded_file.name)
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        # Show uploaded files
        if st.session_state.uploaded_files:
            st.write("📑 Uploaded Documents:")
            for file in st.session_state.uploaded_files:
                st.write(f"- {file}")

def create_chat_interface():
    # Chat title
    st.title("🤖 AI Research Assistant")

    # Chat messages container with custom styling
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your research..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    events = invoke(st.session_state.agent, st.session_state.config, prompt)
                    for event in events:
                        assistant_message = {
                            "role": "assistant",
                            "content": event['messages'][-1].content,
                        }

                    # Add assistant response to messages
                    st.session_state.messages.append(assistant_message)

                    st.write(assistant_message["content"])

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Add custom CSS
    st.markdown("""
        <style>
        .stChat {
            padding: 20px;
        }
        .stChatMessage {
            margin: 10px 0;
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create sidebar for document upload
    create_sidebar()

    # Create main chat interface
    create_chat_interface()

if __name__ == "__main__":
    main()