"""
Model Selection and Serialization Script.

This script uses MLflowClient to query registered models, compare them,
select the best one based on validation accuracy, and serialize it to ONNX format.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient


@click.command()
@click.option("--model-name", default="pet-classifier", help="Registered model name in MLflow")
@click.option("--output-dir", default="./", help="Output directory for serialized model")
@click.option("--metric", default="final_val_accuracy", help="Metric to use for comparison")
def select_and_serialize(model_name, output_dir, metric):
    """
    Select the best model from MLflow registry and serialize to ONNX.

    Args:
        model_name: Name of the registered model in MLflow
        output_dir: Directory to save the serialized model
        metric: Metric name to use for model comparison
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize MLflow client
    client = MlflowClient()

    print(f"Searching for registered models with name: {model_name}")

    # Search for all versions of the registered model
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"Error searching for model versions: {e}")
        print("Make sure you have trained and registered models first.")
        return

    if not model_versions:
        print(f"No model versions found for '{model_name}'")
        print("Please train and register models first using scripts/train.py")
        return

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
        print(f"  Model: {metrics.get('model_name', 'N/A')}")
        print(f"  Epochs: {int(metrics.get('epochs', 0)) if 'epochs' in metrics else 'N/A'}")
        print()

        # Update best model if this one is better
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_version = version

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

    # Download and save class labels
    print("\nDownloading class labels...")
    class_labels_artifact = client.download_artifacts(
        best_version.run_id, "class_labels.json"
    )

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
