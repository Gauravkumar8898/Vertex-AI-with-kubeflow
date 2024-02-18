from src.house_price_pipeline.pipeline import kube_pipeline
from kfp import compiler
from src.utils.constant import pipeline_job_name, template_path, project, location, package_path
from google.cloud import aiplatform


def compile_component_and_run_pipeline():
    """
    Compiles the specified Kubeflow pipeline, initializes AI Platform, and runs the pipeline job.

    This function compiles the Kubeflow pipeline specified by `kube_pipeline` using the AI Platform compiler.
    It then initializes AI Platform using the provided project and location information.
    Finally, it runs the compiled pipeline job with the specified display name and template path.
    """
    compiler.Compiler().compile(kube_pipeline, package_path=package_path)  # Compile the Kubeflow pipeline
    aiplatform.init(project=project, location=location)  # Initialize AI Platform
    job = aiplatform.PipelineJob(
        display_name=pipeline_job_name,  # Display name of the pipeline job
        template_path=template_path  # Path to the pipeline template
    )
    job.run()  # Run the pipeline job


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compile_component_and_run_pipeline()
