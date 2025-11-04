import asyncio
import gradio as gr

from .researchers import default_researchers


async def run_all(topic: str):
    researchers = default_researchers()
    tasks = [asyncio.create_task(r.run(topic)) for r in researchers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs = []
    for name, res in zip([r.name for r in researchers], results):
        if isinstance(res, Exception):
            outputs.append(f"{name}: error: {res}")
        else:
            outputs.append(f"{name}: {res}")
    return outputs


def create_ui():
    with gr.Blocks(title="Asset Analyzer") as ui:
        gr.Markdown("## Asset Analyzer - 4 Researchers")
        topic = gr.Textbox(value="AI and Semiconductors", label="Topic")
        # Build once to derive names for UI labels
        names = [r.name for r in default_researchers()]
        outputs = [gr.Textbox(label=label) for label in names]

        async def _run(topic_val):
            results = await run_all(topic_val)
            # pad to four
            results = (results + [""] * 4)[:4]
            return results

        go = gr.Button("Analyze", variant="primary")
        go.click(fn=_run, inputs=[topic], outputs=outputs)

        # Also run once on load
        ui.load(fn=_run, inputs=[topic], outputs=outputs)
    return ui


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True) #inbrowser=True


