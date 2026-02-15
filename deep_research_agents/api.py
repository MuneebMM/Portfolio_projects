import json
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import config
from graph import research_graph
from database import init_db, save_report, get_report, list_reports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deep Researcher Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str


@app.on_event("startup")
def startup():
    init_db()


@app.post("/research")
async def research(request: ResearchRequest):
    """Start research on a topic. Returns a streaming response."""

    def generate():
        topic = request.topic
        state = {
            "topic": topic,
            "research_findings": "",
            "analysis": "",
            "report": "",
        }

        final_state = {**state}
        for event in research_graph.stream(state):
            for node_name, node_output in event.items():
                final_state.update(node_output)
                yield json.dumps({"node": node_name, "data": node_output}) + "\n"

        if final_state.get("report"):
            report_id = save_report(topic, final_state["report"])
            yield json.dumps({"node": "saved", "report_id": report_id}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/reports")
def get_reports():
    """List all saved research reports."""
    return list_reports()


@app.get("/reports/{report_id}")
def get_report_by_id(report_id: int):
    """Get a specific report by ID."""
    report = get_report(report_id)
    if not report:
        return {"error": "Report not found"}
    return report


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
