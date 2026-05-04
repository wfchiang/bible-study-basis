import asyncio

from langchain.messages import AIMessage
import streamlit as st

from agent import create_agent

# Page configuration
st.set_page_config(
    page_title="Bible Study Basis",
    page_icon="✞",
    layout="wide")
st.title("✞ Bible Study Basis")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the Bible..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status_container = st.status("Thinking...", expanded=True)
        message_placeholder = st.empty()

        async def stream_chat():
            agent = await create_agent()
            inputs = {"messages": st.session_state.messages}
            final_response = "(thinking...)"
            
            async for event in agent.astream_events(inputs, version="v2"):
                event_type = event["event"]
                
                if event_type == "on_tool_start":
                    status_container.write(f"**Calling tool:** `{event['name']}`")
                elif event_type == "on_tool_end":
                    status_container.write("**Tool output:**")
                    status_container.write(event["data"].get("output"))
                elif (event_type == "on_chain_end" and "data" in event
                      and "output" in event["data"] and "messages" in event["data"]["output"]
                      and len(event["data"]["output"]["messages"]) > 0
                      and isinstance(event["data"]["output"]["messages"][-1], AIMessage)):
                    final_response = event["data"]["output"]["messages"][-1].content
            
            status_container.update(label="Finished thinking", state="complete", expanded=False)
            message_placeholder.markdown(final_response)
            return final_response

        try:
            response_content = asyncio.run(stream_chat())
            st.session_state.messages.append({"role": "assistant", "content": response_content})
        except Exception as e:
            st.error(f"An error occurred: {e}")