import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from demo_scenarios import run_priya_scenario, run_untrained_scenario


ACTION_COLORS = {
    "OBSERVE_QUIETLY": "#95a5a6",
    "GENTLE_AWARENESS": "#3498db",
    "PARENT_CHECK_IN": "#f39c12",
    "URGENT_SUPPORT": "#e74c3c",
}

STATE_NUMERIC = {
    "SAFE": 0,
    "VULNERABLE": 1,
    "AT_RISK": 2,
    "IN_DANGER": 3,
}


def build_chart(records, title):
    days = [r["day"] for r in records]
    trust = [r["trust"] for r in records]
    sentiment = [r["sentiment"] for r in records]
    hidden_numeric = [STATE_NUMERIC[r["hidden_state"]] for r in records]
    hidden_labels = [r["hidden_state"] for r in records]
    actions = [r["action"] for r in records]
    rewards = [r["reward"] for r in records]
    cumulative_reward = []
    running = 0
    for r in rewards:
        running += r
        cumulative_reward.append(round(running, 2))

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Guardian Trust Level",
            "Child Sentiment Score",
            "True Risk State (Hidden from Agent)",
            "Cumulative Reward",
        ),
        vertical_spacing=0.10,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    # --- Trust ---
    fig.add_trace(
        go.Scatter(
            x=days, y=trust,
            mode="lines+markers",
            name="Guardian Trust",
            line=dict(color="#2ecc71", width=2),
            marker=dict(size=5),
        ), row=1, col=1
    )
    fig.add_hline(y=0.3, line_dash="dot", line_color="red",
                  annotation_text="Alert ignored below here",
                  annotation_position="bottom right", row=1, col=1)

    # --- Sentiment ---
    fig.add_trace(
        go.Scatter(
            x=days, y=sentiment,
            mode="lines",
            name="Sentiment",
            line=dict(color="#3498db", width=2),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.1)",
        ), row=2, col=1
    )

    # --- Hidden state ---
    state_colors_line = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    fig.add_trace(
        go.Scatter(
            x=days, y=hidden_numeric,
            mode="lines+markers",
            name="Risk State",
            line=dict(color="#e74c3c", width=2),
            marker=dict(
                size=8,
                color=[STATE_NUMERIC[r["hidden_state"]] for r in records],
                colorscale=[[0, "#2ecc71"], [0.33, "#f1c40f"],
                             [0.66, "#e67e22"], [1, "#e74c3c"]],
                showscale=False,
            ),
            text=hidden_labels,
            hovertemplate="Day %{x}: %{text}<extra></extra>",
        ), row=3, col=1
    )
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=["SAFE", "VULNERABLE", "AT_RISK", "IN_DANGER"],
        row=3, col=1
    )

    # --- Cumulative reward ---
    fig.add_trace(
        go.Scatter(
            x=days, y=cumulative_reward,
            mode="lines",
            name="Cumulative Reward",
            line=dict(color="#9b59b6", width=2),
            fill="tozeroy",
            fillcolor="rgba(155,89,182,0.1)",
        ), row=4, col=1
    )

    # --- Intervention markers on trust chart ---
    for r in records:
        if r["action"] != "OBSERVE_QUIETLY":
            fig.add_vline(
                x=r["day"],
                line_dash="dash",
                line_color=ACTION_COLORS.get(r["action"], "gray"),
                line_width=1,
                row=1, col=1,
            )

    fig.update_layout(
        height=750,
        title_text=title,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(l=60, r=40, t=80, b=40),
    )
    fig.update_xaxes(title_text="Day", row=4, col=1)

    return fig


def build_table(records):
    df = pd.DataFrame([{
        "Day": r["day"],
        "Action": r["action"],
        "Risk State": r["hidden_state"],
        "Guardian Trust": f"{r['trust']:.0%}",
        "Guardian Response": r["guardian_response"],
        "Reward": f"{r['reward']:+.2f}",
    } for r in records])
    return df


def run_trained():
    logs, records = run_priya_scenario()
    total = sum(r["reward"] for r in records)
    final_state = records[-1]["hidden_state"]
    final_trust = records[-1]["trust"]

    summary = (
        f"**Final risk state:** {final_state}  \n"
        f"**Final guardian trust:** {final_trust:.0%}  \n"
        f"**Total reward:** {total:.2f}  \n"
        f"**Interventions sent:** "
        f"{sum(1 for r in records if r['action'] != 'OBSERVE_QUIETLY')}"
    )

    fig = build_chart(records, "Trained Agent — Priya's Story")
    df = build_table(records)
    return fig, df, summary


def run_untrained():
    logs, records = run_untrained_scenario()
    total = sum(r["reward"] for r in records)
    final_state = records[-1]["hidden_state"]
    final_trust = records[-1]["trust"]

    summary = (
        f"**Final risk state:** {final_state}  \n"
        f"**Final guardian trust:** {final_trust:.0%}  \n"
        f"**Total reward:** {total:.2f}  \n"
        f"**Interventions sent:** "
        f"{sum(1 for r in records if r['action'] != 'OBSERVE_QUIETLY')}"
    )

    fig = build_chart(records, "Untrained Agent — Alert Fatigue Collapse")
    df = build_table(records)
    return fig, df, summary


with gr.Blocks(title="SafeSignal Demo", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🛡️ SafeSignal")
    gr.Markdown(
        "An RL-trained agent that learns **when to alert parents and when to "
        "stay silent** — preserving the trust that makes future warnings matter."
    )

    with gr.Row():
        trained_btn = gr.Button(
            "▶ Run Trained Agent (Priya's Story)", variant="primary", scale=2
        )
        untrained_btn = gr.Button(
            "▶ Run Untrained Agent (Alert Fatigue)", variant="secondary", scale=2
        )

    gr.Markdown(
    "**Intervention markers:** "
    "<span style='color:#3498db'>■ Gentle Awareness</span> · "
    "<span style='color:#f39c12'>■ Parent Check-In</span> · "
    "<span style='color:#e74c3c'>■ Urgent Support</span>"
    )

    summary_box = gr.Markdown()
    chart = gr.Plot(label="30-Day Behavioral Timeline")
    table = gr.Dataframe(label="Day-by-Day Log", wrap=True)

    trained_btn.click(
        fn=run_trained,
        outputs=[chart, table, summary_box]
    )
    untrained_btn.click(
        fn=run_untrained,
        outputs=[chart, table, summary_box]
    )


if __name__ == "__main__":
    app.launch(inbrowser=True)