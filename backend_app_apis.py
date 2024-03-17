from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
import json
import os
import logging
from datetime import datetime
from lm_eval import tasks, evaluator, utils

logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI()

@app.get("/check/")
async def check():
  return "Hello World, it's working"


@app.post("/run_evaluation/")
async def run_evaluation_api(eval_request):
    task_names = ["medmcqa", "medqa_4options", "mmlu_anatomy", "mmlu_clinical_knowledge", "mmlu_college_biology", "mmlu_college_medicine", "mmlu_medical_genetics", "mmlu_professional_medicine", "pubmedqa"]
    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=eval_request,
            tasks=task_names,
            batch_size="auto",
            device="cuda:0",
            limit=None,
            write_out=True
            )
        results_ = {'status': True, 'result': results}
  
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {'status': False, 'error': str(e)}
    
    return results_


if __name__ == "__main__":
    uvicorn.run("run_app:app", host="127.0.0.1", port=8000, log_level="info")
