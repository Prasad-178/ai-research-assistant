import streamlit as st
from utils import extract_text_from_pdf, index_document
from agent import create_agent, AgentState
import uuid

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = create_agent()

def main():
    st.title("AI Research Assistant")
    
    init_session_state()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:
        with st.spinner("Processing document..."):
            text = extract_text_from_pdf(uploaded_file)
            doc_id = str(uuid.uuid4())
            index_document(text, metadata={"doc_id": doc_id, "filename": uploaded_file.name})
            st.success("Document processed and indexed successfully!")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me anything about your research..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepare agent state
        state = AgentState(
            messages=st.session_state.messages,
            current_message=prompt,
            context=[],
            tool_calls=[]
        )
        
        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                final_state = st.session_state.agent(state)
                st.session_state.messages = final_state["messages"]
                # Display the last assistant message
                for msg in reversed(final_state["messages"]):
                    if msg["role"] == "assistant":
                        st.write(msg["content"])
                        break

if __name__ == "__main__":
    main()