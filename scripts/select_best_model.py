"""
Model Selection and Serialization Script.

This script uses MLflowClient to query registered models, compare them,
select the best one based on validation accuracy, and serialize it to ONNX format.
"""

import json
from pathlib import Path

import click
import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient


@click.command()
@click.option("--output-dir", default="./", help="Output directory for serialized model")
@click.option("--metric", default="final_val_accuracy", help="Metric to use for comparison")
def select_and_serialize(output_dir, metric):
    """
    Select the best model from MLflow registry and serialize to ONNX.
    Searches across all pet-classifier-* models and selects based on size priority.

    Args:
        output_dir: Directory to save the serialized model
        metric: Metric name to use for model comparison
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize MLflow client
    client = MlflowClient()

    print(f"Searching for all registered pet-classifier models...")

    # Search for ALL registered models
    try:
        all_models = client.search_registered_models()
        # Filter for pet-classifier models
        pet_models = [m for m in all_models if m.name.startswith("pet-classifier-")]
    except Exception as e:
        print(f"Error searching for models: {e}")
        print("Make sure you have trained and registered models first.")
        return

    if not pet_models:
        print(f"No pet-classifier models found")
        print("Please train and register models first using scripts/train.py")
        return
    
    print(f"Found {len(pet_models)} pet-classifier models")
    
    # Get all versions from all models
    model_versions = []
    for model in pet_models:
        versions = client.search_model_versions(f"name='{model.name}'")
        model_versions.extend(versions)

    print(f"Found {len(model_versions)} model version(s)")

    # Compare models and select the best one
    best_version = None
    best_metric_value = -1.0

    print(f"\nComparing models based on '{metric}':")
    print("-" * 80)

    for version in model_versions:
        run_id = version.run_id
        version_number = version.version

        # Get run details
        run = client.get_run(run_id)
        metrics = run.data.metrics

        # Get the comparison metric
        metric_value = metrics.get(metric, -1.0)

        print(f"Version {version_number}:")
        print(f"  Run ID: {run_id}")
        print(f"  {metric}: {metric_value:.2f}%")
        print(f"  Model: {run.data.params.get('model_name', 'N/A')}")
        print(f"  Epochs: {run.data.params.get('epochs', 'N/A')}")
        print()

        # Update best model based on SIZE (lightest first), then accuracy
        # Define model size priority (smaller index = lighter/better)
        size_priority = {
            "mobilenet_v2": 0,
            "efficientnet_b0": 1,
            "resnet18": 2,
            "resnet50": 3,
            "vgg16": 4
        }
        
        current_model_name = run.data.params.get('model_name', 'unknown')
        
        # If we haven't selected a model yet, take this one
        if best_version is None:
            best_version = version
            best_metric_value = metric_value
            continue
            
        # Get details of the currently selected best model
        best_run = client.get_run(best_version.run_id)
        best_model_name = best_run.data.params.get('model_name', 'unknown')
        
        current_priority = size_priority.get(current_model_name, 99)
        best_priority = size_priority.get(best_model_name, 99)
        
        # Logic:
        # 1. If current model is lighter (lower priority index), select it
        # 2. If models have same size priority, select the one with higher accuracy
        if current_priority < best_priority:
            print(f"  -> Selecting {current_model_name} over {best_model_name} (Lighter: {current_priority} < {best_priority})")
            best_version = version
            best_metric_value = metric_value
        elif current_priority == best_priority:
            if metric_value > best_metric_value:
                print(f"  -> Selecting {current_model_name} (Higher Accuracy: {metric_value:.2f} > {best_metric_value:.2f})")
                best_version = version
                best_metric_value = metric_value

    if best_version is None:
        print("Could not find a best model")
        return

    print("=" * 80)
    print(f"✓ Best model: Version {best_version.version}")
    print(f"  {metric}: {best_metric_value:.2f}%")
    print(f"  Run ID: {best_version.run_id}")
    print("=" * 80)

    # Load the best model
    print("\nLoading best model from MLflow...")
    model_uri = f"runs:/{best_version.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)

    # Move model to CPU (required for Render deployment)
    print("Moving model to CPU...")
    model = model.to("cpu")

    # Set model to evaluation mode
    model.eval()

    # Serialize model to ONNX format
    print("Serializing model to ONNX format...")
    onnx_path = output_path / "model.onnx"

    # Create dummy input for ONNX export (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    # Export to ONNX - embed all data in single file (no external .data file)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"✓ Model serialized to: {onnx_path.absolute()}")

    # Download and save class labels - AUTO-DETECT JSON FILE
    print("\nDownloading class labels...")
    
    # List artifacts to find class_labels JSON file
    artifacts = client.list_artifacts(best_version.run_id)
    
    # Find any JSON file containing "class_labels"
    json_candidates = [
        art.path
        for art in artifacts
        if "class_labels" in art.path and art.path.endswith(".json")
    ]

    if not json_candidates:
        print("Warning: No class_labels JSON file found in artifacts")
        print("Available artifacts:")
        for art in artifacts:
            print(f"  - {art.path}")
        return

    # Use the first match
    artifact_path = json_candidates[0]
    print(f"Found class labels artifact: {artifact_path}")

    class_labels_artifact = client.download_artifacts(best_version.run_id, artifact_path)

    # Load the class labels
    with open(class_labels_artifact, "r", encoding="utf-8") as f:
        class_labels = json.load(f)

    # Save class labels to output directory
    class_labels_path = output_path / "class_labels.json"
    with open(class_labels_path, "w", encoding="utf-8") as f:
        json.dump(class_labels, f, indent=2)

    print(f"✓ Class labels saved to: {class_labels_path.absolute()}")

    # Print summary
    print("\n" + "=" * 80)
    print("MODEL SERIALIZATION COMPLETE")
    print("=" * 80)
    print(f"Model file: {onnx_path.absolute()}")
    print(f"Class labels: {class_labels_path.absolute()}")
    print(f"Number of classes: {len(class_labels)}")
    print(f"Best validation accuracy: {best_metric_value:.2f}%")
    print("=" * 80)

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")


if __name__ == "__main__":
    select_and_serialize()
