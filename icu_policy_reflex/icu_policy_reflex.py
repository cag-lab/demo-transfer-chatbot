"""Reflex interface for the ICU Transfer Policy Assistant."""

from __future__ import annotations

import reflex as rx

from rag import Index, answer

_index: Index | None = None


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index()
    return _index


class ChatState(rx.State):
    messages: list[dict[str, str]] = []
    question: str = ""
    indexed_passages: int = 0
    is_loading: bool = False
    error: str = ""

    @rx.event
    def load(self) -> None:
        try:
            index = get_index()
            self.indexed_passages = len(index.chunks)
            self.error = ""
        except Exception as exc:  # pragma: no cover - UI fallback
            self.error = str(exc)

    @rx.event
    def clear_conversation(self) -> None:
        self.messages = []

    @rx.event
    def set_question(self, value: str) -> None:
        self.question = value

    @rx.event
    def ask(self):
        prompt = self.question.strip()
        if not prompt:
            return

        self.messages.append({"role": "user", "content": prompt, "sources": ""})
        self.question = ""
        self.is_loading = True
        yield

        try:
            index = get_index()
            history = [{"role": m["role"], "content": m["content"]} for m in self.messages[:-1]]
            reply, hits = answer(prompt, index, history)
            sources = "\n".join(
                f"**[{n}]** {chunk.title} — *{chunk.section}* (`{chunk.source}`, relevance {score:.2f})\n\n> {chunk.text[:400]}…"
                for n, (chunk, score) in enumerate(hits, start=1)
            )
            self.messages.append({"role": "assistant", "content": reply, "sources": sources})
            self.indexed_passages = len(index.chunks)
            self.error = ""
        except Exception as exc:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": f"Something went wrong calling Azure OpenAI: {exc}",
                    "sources": "",
                }
            )
        finally:
            self.is_loading = False


def chat_message(message: rx.Var[dict[str, str]]) -> rx.Component:
    role_color = rx.cond(message["role"] == "assistant", "blue", "gray")
    return rx.box(
        rx.badge(message["role"], color_scheme=role_color),
        rx.markdown(message["content"], margin_top="0.5rem"),
        rx.cond(
            message["sources"] != "",
            rx.accordion.root(
                rx.accordion.item(
                    header="Sources",
                    content=rx.markdown(message["sources"]),
                    value="sources",
                ),
                collapsible=True,
                type="single",
                width="100%",
                margin_top="0.5rem",
            ),
        ),
        width="100%",
        border="1px solid var(--gray-5)",
        border_radius="12px",
        padding="1rem",
        background=rx.cond(message["role"] == "assistant", "var(--accent-2)", "white"),
    )


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("ICU Transfer Policy Assistant", size="7"),
            rx.text(
                "Answers come only from your policy documents. Policies support, but never replace, clinical judgment.",
                color="var(--gray-11)",
            ),
            rx.cond(
                ChatState.error != "",
                rx.callout(
                    ChatState.error,
                    icon="triangle_alert",
                    color_scheme="red",
                    width="100%",
                ),
            ),
            rx.hstack(
                rx.button("Clear conversation", on_click=ChatState.clear_conversation, variant="soft"),
                rx.badge(rx.text("Passages indexed: "), rx.text(ChatState.indexed_passages), color_scheme="green"),
                width="100%",
                justify="between",
                align="center",
            ),
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(ChatState.messages, chat_message),
                    spacing="3",
                    width="100%",
                ),
                type="always",
                scrollbars="vertical",
                style={"height": "58vh", "width": "100%", "paddingRight": "8px"},
            ),
            rx.hstack(
                rx.input(
                    placeholder="Ask about ICU transfer policy…",
                    value=ChatState.question,
                    on_change=ChatState.set_question,
                    width="100%",
                ),
                rx.button("Send", on_click=ChatState.ask, loading=ChatState.is_loading),
                width="100%",
            ),
            spacing="4",
            width="100%",
            max_width="900px",
            margin_x="auto",
            padding_y="2rem",
        ),
    )


app = rx.App(theme=rx.theme(appearance="light", accent_color="blue"))
app.add_page(index, title="ICU Transfer Policy Assistant", on_load=ChatState.load)
