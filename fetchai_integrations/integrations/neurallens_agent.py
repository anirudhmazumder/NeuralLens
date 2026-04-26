"""
integrations/neurallens_agent.py

Fetch.ai uAgent wrapper for NeuralLens.
Publishes the pipeline as a discoverable agent on Agentverse / ASI:One.

Run with:
    python integrations/neurallens_agent.py

The agent prints its address on startup — paste that into Agentverse to
register it under the neurallens manifest.
"""

# stdlib
import re
from typing import Optional

# third-party
from uagents import Agent, Context, Model
from uagents.experimental.chat import ChatMessage, ChatAcknowledgement, chat_protocol  # type: ignore

# local
from pipeline.optimization_loop import run_full_pipeline

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class AnalysisRequest(Model):
    """Direct programmatic request model (non-chat path)."""
    url: str
    industry: str = "general"


class AnalysisResponse(Model):
    """Response payload for direct AnalysisRequest messages."""
    nes_before: float
    nes_after: float
    delta: float
    profile_before: str
    profile_after: str
    summary: str
    overlay_before_url: str
    overlay_after_url: str
    screenshot_before_url: str
    screenshot_after_url: str


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

agent = Agent(
    name="neurallens",
    seed="neurallens_lahacks_2026",
    port=8000,
    endpoint=["http://localhost:8000/submit"],
)


@agent.on_event("startup")
async def on_startup(ctx: Context) -> None:
    """Print the agent address so it can be registered on Agentverse."""
    ctx.logger.info(f"NeuralLens agent address: {agent.address}")
    print(f"\n[NeuralLens] Agent address: {agent.address}\n")


# ---------------------------------------------------------------------------
# Chat protocol (ASI:One / Agentverse chat)
# ---------------------------------------------------------------------------

chat_proto = chat_protocol(publish_manifest=True)


def _detect_industry(text: str) -> str:
    """Infer business industry from free-text keywords."""
    text_lower = text.lower()
    keywords = {
        "ecommerce": ["shop", "store", "buy", "cart", "product", "sale"],
        "saas": ["software", "app", "platform", "dashboard", "trial", "subscription"],
        "restaurant": ["food", "menu", "eat", "restaurant", "order", "delivery"],
        "healthcare": ["health", "clinic", "doctor", "medical", "patient"],
        "real estate": ["property", "home", "house", "rent", "buy", "real estate"],
        "finance": ["bank", "loan", "invest", "finance", "credit", "money"],
    }
    for industry, words in keywords.items():
        if any(w in text_lower for w in words):
            return industry
    return "general"


def _extract_url(text: str) -> Optional[str]:
    """Extract first HTTP/HTTPS URL from free text."""
    match = re.search(r"https?://\S+", text)
    return match.group() if match else None


def _format_result(result: dict) -> str:
    """Convert pipeline result dict to a readable chat response."""
    delta_sign = "+" if result["delta"] >= 0 else ""
    changes_text = "\n".join(
        f"  {i+1}. [{c.get('target_region', '?')}] {c.get('reason', '')}"
        for i, c in enumerate(result.get("changes", []))
    )
    return (
        f"NeuralLens Analysis Complete\n"
        f"{'─' * 40}\n"
        f"URL: {result['url']}\n"
        f"Industry: {result['industry']}\n\n"
        f"NES Before:  {result['nes_before']}/100  ({result['profile_before']})\n"
        f"NES After:   {result['nes_after']}/100  ({result['profile_after']})\n"
        f"Improvement: {delta_sign}{result['delta']} points\n\n"
        f"Optimisation strategy:\n{result['summary']}\n\n"
        f"Brain-targeted changes:\n{changes_text}\n\n"
        f"Before overlay: {result['overlay_before_url']}\n"
        f"After overlay:  {result['overlay_after_url']}\n"
        f"Before screenshot: {result['screenshot_before_url']}\n"
        f"After screenshot:  {result['screenshot_after_url']}\n"
    )


@chat_proto.on_message(ChatMessage)
async def handle_chat(ctx: Context, sender: str, message: ChatMessage) -> None:
    """Handle free-text chat messages from ASI:One or Agentverse."""
    # Acknowledge immediately
    await ctx.send(sender, ChatAcknowledgement(acknowledged=True))

    content = message.content if isinstance(message.content, str) else str(message.content)
    url = _extract_url(content)

    if not url:
        await ctx.send(
            sender,
            ChatMessage(content="Please include a URL (starting with http:// or https://) in your message."),
        )
        return

    industry = _detect_industry(content)

    await ctx.send(
        sender,
        ChatMessage(content=f"Analysing {url} (industry: {industry})... this takes ~30 seconds."),
    )

    try:
        result = run_full_pipeline(url, industry)
        reply = _format_result(result)
    except Exception as exc:
        reply = f"Analysis failed: {exc}"

    await ctx.send(sender, ChatMessage(content=reply))


# ---------------------------------------------------------------------------
# Direct request/response protocol
# ---------------------------------------------------------------------------

@agent.on_message(AnalysisRequest, replies={AnalysisResponse})
async def handle_analysis_request(ctx: Context, sender: str, msg: AnalysisRequest) -> None:
    """Handle direct AnalysisRequest messages (programmatic callers)."""
    ctx.logger.info(f"Received AnalysisRequest for {msg.url}")
    result = run_full_pipeline(msg.url, msg.industry)

    await ctx.send(
        sender,
        AnalysisResponse(
            nes_before=result["nes_before"],
            nes_after=result["nes_after"],
            delta=result["delta"],
            profile_before=result["profile_before"],
            profile_after=result["profile_after"],
            summary=result["summary"],
            overlay_before_url=result["overlay_before_url"],
            overlay_after_url=result["overlay_after_url"],
            screenshot_before_url=result["screenshot_before_url"],
            screenshot_after_url=result["screenshot_after_url"],
        ),
    )


agent.include(chat_proto)

if __name__ == "__main__":
    agent.run()
