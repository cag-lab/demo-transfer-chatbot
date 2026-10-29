"""Local chat interface.  Start with:  streamlit run app.py"""
import streamlit as st

from rag import Index, answer

st.set_page_config(page_title="ICU Transfer Policy Assistant", page_icon="🏥")
st.title("ICU Transfer Policy Assistant")
st.caption("Answers come only from the documents in the ./docs folder. "
           "Policies support, but never replace, clinical judgment.")


@st.cache_resource
def load_index():
    return Index()


try:
    index = load_index()
except Exception as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(f"**Passages indexed:** {len(index.chunks)}")
    st.markdown("Nothing is saved to disk. Closing the tab ends the conversation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.markdown(msg["sources"])

if question := st.chat_input("Ask about ICU transfer policy…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies…"):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            try:
                reply, hits = answer(question, index, history)
            except Exception as e:
                reply, hits = f"Something went wrong calling Azure OpenAI: {e}", []
        st.markdown(reply)
        sources = "\n".join(
            f"**[{n}]** {c.title} — *{c.section}* (`{c.source}`, relevance {score:.2f})\n\n> {c.text[:400]}…"
            for n, (c, score) in enumerate(hits, start=1)
        )
        if sources:
            with st.expander("Sources"):
                st.markdown(sources)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
