"""
GCP Vertex AI Pipeline Trigger Client.
Demonstrates closed-loop automated MLOps retraining when edge covariate drift is detected.
Dispatches a Vertex AI PipelineJob or Custom Training Run via Google Cloud Python SDK / REST.
"""

import argparse
import json
import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VertexTrigger")


def trigger_vertex_pipeline(
    project_id: str,
    region: str,
    pipeline_name: str,
    dataset_uri: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Submits a retraining pipeline to GCP Vertex AI Pipelines.
    In cloud environments with google-cloud-aiplatform installed and authenticated,
    this initializes aiplatform and creates a PipelineJob.
    """
    logger.info(f"Connecting to GCP Vertex AI in project '{project_id}', region '{region}'...")
    logger.info(f"Target Retraining Pipeline: {pipeline_name}")
    logger.info(f"Covariate Drift Dataset URI: {dataset_uri}")

    payload = {
        "pipeline_spec_uri": f"gs://{project_id}-pipelines/{pipeline_name}.json",
        "pipeline_root": f"gs://{project_id}-pipeline-artifacts/",
        "parameter_values": {
            "dataset_uri": dataset_uri,
            "drift_severity": parameters.get("drift_severity", "HIGH"),
            "epochs": parameters.get("epochs", 30),
            "batch_size": parameters.get("batch_size", 64),
        },
    }

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project_id, location=region)
        job = aiplatform.PipelineJob(
            display_name=f"{pipeline_name}-drift-retrain",
            template_path=payload["pipeline_spec_uri"],
            pipeline_root=payload["pipeline_root"],
            parameter_values=payload["parameter_values"],
        )
        job.submit()
        logger.info(f"[SUCCESS] Vertex AI PipelineJob submitted successfully: {job.resource_name}")
        return {"status": "SUBMITTED", "job_id": job.resource_name}
    except ImportError:
        logger.warning(
            "google-cloud-aiplatform library not installed locally. "
            "Simulating successful cloud dispatch payload for local edge deployment."
        )
        return {
            "status": "DISPATCH_PAYLOAD_PREPARED",
            "gcp_project": project_id,
            "gcp_region": region,
            "pipeline_payload": payload,
        }


def main():
    parser = argparse.ArgumentParser(description="Trigger GCP Vertex AI Retraining Pipeline on Drift")
    parser.add_argument("--project", default="edge-sentinel-mlops-prod", help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--pipeline", default="edge-vision-retrain-pipeline", help="Pipeline name")
    parser.add_argument("--dataset", default="gs://edge-sentinel-mlops-prod/drift-snapshots/edge-01.parquet", help="GCS Dataset URI")
    args = parser.parse_args()

    result = trigger_vertex_pipeline(
        project_id=args.project,
        region=args.region,
        pipeline_name=args.pipeline,
        dataset_uri=args.dataset,
        parameters={"drift_severity": "HIGH", "epochs": 25},
    )
    print("\n--- GCP Vertex AI Trigger Response ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
