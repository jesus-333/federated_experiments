# Define strategy
strategy = FedXgbBagging(
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_evaluate,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    on_evaluate_config_fn=config_func,
    on_fit_config_fn=config_func,
    initial_parameters=parameters,
)


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config
