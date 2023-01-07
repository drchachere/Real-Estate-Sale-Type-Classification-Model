from kedro.pipeline import Pipeline, node

from .nodes import create_X_y, log_reg_models, rand_for_models, neu_net_models

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_X_y,
                inputs=["train", "params:cols"],
                outputs=["X", "y"],
                name="create_X_y_node",
            ),
            node(
                func=log_reg_models,
                inputs=["X", "y"],
                outputs="lr_model_scores",
                name="log_reg_models_node",
            ),
            node(
                func=rand_for_models,
                inputs=["X", "y"],
                outputs="rf_model_scores",
                name="rand_for_models_node",
            ),
            node(
                func=neu_net_models,
                inputs=["X", "y"],
                outputs="nn_model_scores",
                name="neu_net_models_node",
            )
            # node(
            #     func=plot_scatter,
            #     inputs="shuttles",
            #     outputs="preprocessed_shuttles",
            #     name="plot_scatter_node",
            # ),
            # node(
            #     func=standardize_foi,
            #     inputs=["params:cols", "train_with_features"],
            #     outputs="train_stand",
            #     name="standardize_foi_node",
            # ),
            # node(
            #     func=tts,
            #     inputs=["params:cols", "train_stand"],
            #     outputs=["X_train", "X_test", "y_train", "y_test"],
            #     name="tts_node",
            # ),
            # node(
            #     func=find_model_perf,
            #     inputs=["X_train", "y_train", "X_test", "y_test"],
            #     outputs=None,
            #     name="find_model_perf_node",
            # )
        ]
    )