import numpy as np
import requests
import streamlit as st
import os
import json
import mlflow

from mlflow.tracking import MlflowClient
client = MlflowClient()

# @st.cache
def get_instances():

    experiment_name = 'Checker'
    executor_name = 'Checker'
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    query = f"tag.mlflow.runName='{executor_name}'"
    runs_info = client.search_runs(experiment_id, query,
                                   order_by=["attribute.start_time DESC"],
                                   max_results=1)

    run_id = runs_info[0].info.run_id
    client.download_artifacts(runs_info[0].info.run_id,
                              'x_chosen.npy',
                              '/tmp/')

    x_chosen_path = os.path.join('/tmp', 'x_chosen.npy')
    x_chosen = np.load(x_chosen_path)

    return x_chosen, run_id



def main():

    images, run_id = get_instances()

    with st.sidebar:
        st.header("Image Labeling")
        with st.form(key="grid_reset"):
            n_photos = st.slider("Number of images:", 2, images.shape[0], images.shape[0])
            n_cols = st.number_input("Number of columns", 2, 8, 4)
            st.form_submit_button(label="Reload layout")
#         with st.beta_expander("About this app"):
#             st.markdown("Custom Labeling")
        st.caption("Custom labeling app")

    st.title("Please provide labels to the following images")
#     st.caption("You can display the image in full size by hovering it and clicking the double arrow")

    cat_images = images[:n_photos]
    n_rows = 1 + len(cat_images) // n_cols
    rows = [st.beta_container() for _ in range(n_rows)]
    cols_per_row = [r.beta_columns(n_cols) for r in rows]

    for image_index, cat_image in enumerate(cat_images):
        with rows[image_index // n_cols]:
            cols_per_row[image_index // n_cols][image_index % n_cols].image(cat_image, str(image_index))


    labels = st.text_input('Labels: write your labels separated by commas (e.g., 8,9,0 etc)')
    labels = labels.strip().split(',')
    n_labels = len(labels) if labels != [''] else 0

    if st.button('Save labels'):
        if n_labels == images.shape[0]:
            st.write(f'{n_labels} labels provided. Length matches with input.')

            y_labels = np.array([int(e) for e in labels])
            filename = 'y_labels.npy'
            with open(filename, 'wb') as f:
                np.save(f, y_labels)

            with mlflow.start_run(run_name="labeling") as mlrun:
                mlflow.log_artifact(filename)
                mlflow.log_param(key='n_labels',
                                 value=f"{y_labels.shape[0]}")
                mlflow.log_param(key='run_detector_id',
                                 value=f"{run_id}")

            content = {'y_labels_artifact': filename,
                       'run_labeling_id': mlrun.info.run_id,

                       'x_chosen_artifact': 'x_chosen.npy',
                       'run_detector_id': run_id,

                       'n_labels': y_labels.shape[0]}

            checker_ctn = "inference-mnist-checker-agent"
            checker_port = 8004
            checker_agent_base_uri = f"http://{checker_ctn}:{checker_port}"
            checker_agent_api_uri = f"{checker_agent_base_uri}/checker/labeling"
            response = requests.post(
                url=checker_agent_api_uri,
                data = json.dumps(content)
            )
            st.write(response.content)
            t_url = mlflow.get_tracking_uri()
            t_url = f"{t_url.split(':')[0]}://0.0.0.0:{t_url.split(':')[-1]}"
            st.success(f'Labels were saved to repository: {t_url}')
        else:
            st.warning(f'{n_labels} labels provided. The number of labels should match the input.')

if __name__ == '__main__':
    main()
