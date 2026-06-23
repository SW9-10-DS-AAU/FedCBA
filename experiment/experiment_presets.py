from dataclasses import dataclass

# Parameters controlled by the default preset
@dataclass(frozen=True)
class DefaultPreset:
    fork: bool
    reward: int
    standard_buy_in: int
    min_buy_in: int
    max_buy_in: int
    first_round_fee: int
    punish_factor: int
    punish_factor_contrib: int
    force_merge_all: bool
    use_nobody_is_kicked: bool
    number_of_inactive_contributors: int


# Parameters specific to experiments
@dataclass(frozen=True)
class ExperimentPreset:
    number_of_good_contributors: int
    number_of_bad_contributors: int
    number_of_freerider_contributors: int
    minimum_rounds: int
    epochs: int
    batch_size: int
    use_outlier_detection: list[bool]
    contribution_score_strategy: list[str]
    freerider_noise_scale: list[float] | None
    freerider_start_round: list[int] | None
    freerider_attack_type: list[str] | None
    malicious_noise_scale: list[float] | None
    malicious_start_round: list[int] | None
    malicious_attack_type: list[str] | None
    aggregation_rule: list[str]
    data_distribution: list[str]
    dirichlet_alpha: list[float] | None
    number_of_runs: int


# Full preset (used when use_defaults=False)
@dataclass(frozen=True)
class FullPreset(DefaultPreset, ExperimentPreset):
    pass


PRESETS = {
    "default": DefaultPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=0,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=False,
        use_nobody_is_kicked=False,
        number_of_inactive_contributors=0,
    ),

    "test": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0],
        freerider_start_round=[2],
        freerider_attack_type=["delta_weight"],
        malicious_noise_scale=[0.1],
        malicious_start_round=[2],
        malicious_attack_type=["byzantine"],
        aggregation_rule=["partial_switch[retro,positives_only,FedAVG]"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=3
    ),


    "mnist_openfl_low_noise": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive"],
        freerider_noise_scale=[0.01],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "mnist_openfl_high_noise": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive"],
        freerider_noise_scale=[0.1],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "cifar_openfl_low_noise": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive"],
        freerider_noise_scale=[0.01],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "cifar_openfl_high_noise": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive"],
        freerider_noise_scale=[0.1],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "mnist_openfl_w_outlier": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive"],
        freerider_noise_scale=[0.1],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        malicious_start_round=[1, 3, 5],
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "mnist_openfl_w/o_outlier": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[False],
        contribution_score_strategy=["dotproduct"],
        freerider_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        malicious_start_round=[1, 3, 5],
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=1
    ),

    "cifar_openfl_w_outlier": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only", "accuracy_loss", "naive", "dotproduct"],
        freerider_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=1
    ),

    "cifar_openfl_w/o_outlier": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[False],
        contribution_score_strategy=["dotproduct"],
        freerider_noise_scale=[0, 0.01, 0.1, 0.5, 1.0],
        freerider_start_round=[1, 3, 5],
        freerider_attack_type=None,
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=1
    ),

    "aggregation_rules_test_model_performance_mnist": FullPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=True,
        use_nobody_is_kicked=True,
        number_of_inactive_contributors=0,
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=5,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only", "accuracy_only"],
        freerider_noise_scale=[0, 0.1, 1.0],
        freerider_start_round=[1, 5, 10],
        freerider_attack_type=None,
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["positives_only", "FedAVG", "plus_one_normalize"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=1
    ),

    "aggregation_rules_test_model_performance_people_get_kicked_now_mnist": FullPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=False,
        use_nobody_is_kicked=False,
        number_of_inactive_contributors=0,
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],  # loss_only is the only loss'os
        freerider_noise_scale=[0.01],               # 0.0
        freerider_start_round=[1],                  # 1
        freerider_attack_type=None,
        malicious_noise_scale=[0.1],
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["GRS_aggregation", "FedAVG", "positives_only", "binary_switch", "plus_one_normalize"],
        data_distribution=["random_split"],         # 1
        dirichlet_alpha=None,
        number_of_runs=1
    ),

    "aggregation_rules_test_model_performance_people_get_kicked_now_cifar": FullPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=False,
        use_nobody_is_kicked=True,
        number_of_inactive_contributors=0,
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=50,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],  # loss_only is the only loss'os
        freerider_noise_scale=[0.01],               # 0.0
        freerider_start_round=[1],                  # 1
        freerider_attack_type=None,
        malicious_noise_scale=[0.1],
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["binary_switch", "positives_only", "FedAVG", "plus_one_normalize"],
        data_distribution=["random_split"],         # 1
        dirichlet_alpha=None,
        number_of_runs=1
    ),

    "data_distribution_mnist": FullPreset(
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=True,
        use_nobody_is_kicked=True,
        number_of_inactive_contributors=0,
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=50,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],  # loss_only is the only loss'os
        freerider_noise_scale=[0.0],                # 0.0
        freerider_start_round=[1],                  # 1
        freerider_attack_type=None,
        malicious_noise_scale=[0.1],
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["positives_only", "FedAVG", "plus_one_normalize"],
        data_distribution=["random_split", "stratified_split", "dirichlet_split"],
        dirichlet_alpha=[0.5, 5.0],
        number_of_runs=1
    ),

    "p10_freerider_noise_mnist": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=2,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0.001, 0.01, 0.1, 1, 10],
        freerider_start_round=[3],
        freerider_attack_type=["noise"],
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split_42"],
        dirichlet_alpha=None,
        number_of_runs=10,
    ),

    "p10_freerider_noise_cifar": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=2,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0.001, 0.01, 0.1, 1, 10],
        freerider_start_round=[3],
        freerider_attack_type=["noise"],
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split_42"],
        dirichlet_alpha=None,
        number_of_runs=10,
    ),

    "p10_freerider_delta_weight_mnist": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=2,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0.0],
        freerider_start_round=[3],
        freerider_attack_type=["delta_weight"],
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split_42"],
        dirichlet_alpha=None,
        number_of_runs=10,
    ),

    "p10_freerider_delta_weight_cifar": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=2,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0.0],
        freerider_start_round=[3],
        freerider_attack_type=["delta_weight"],
        malicious_noise_scale=None,
        malicious_start_round=None,
        malicious_attack_type=None,
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split_42"],
        dirichlet_alpha=None,
        number_of_runs=10,
    ),

    "p10_malicious_noise_byzantine_mnist": ExperimentPreset(
        number_of_good_contributors=4,
        number_of_bad_contributors=2,
        number_of_freerider_contributors=0,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=None,
        freerider_start_round=None,
        freerider_attack_type=None,
        malicious_noise_scale=[0.001, 0.01, 0.1, 1.0, 10.0],
        malicious_start_round=[3],
        malicious_attack_type=["noise", "byzantine"],
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split_42"],
        dirichlet_alpha=None,
        number_of_runs=10,
    ),

    "p10_malicious_noise_byzantine_cifar": ExperimentPreset(
        number_of_good_contributors=6,
        number_of_bad_contributors=2,
        number_of_freerider_contributors=0,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=None,
        freerider_start_round=None,
        freerider_attack_type=None,
        malicious_noise_scale=[0.001, 0.01, 0.1, 1.0, 10.0],
        malicious_start_round=[3],
        malicious_attack_type=["noise", "byzantine"],
        aggregation_rule=["FedAVG"],
        data_distribution=["random_split_42"],
        dirichlet_alpha=None,
        number_of_runs=10,
    ),

    "p10_fedavg_vs_grs_freerider_mnist": FullPreset(
        fork=True,
        reward=int(0),
        # No reward — all users get the same reward each round anyway, so it's simplest to leave their GRS at 1.1 and 0.8.
        standard_buy_in=int(0.8e18),  # bad user's starting GRS
        min_buy_in=int(0.8e18),  # bad user's starting GRS
        max_buy_in=int(1.1e18),  # god brugers start-GRS
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=True,
        use_nobody_is_kicked=False,
        number_of_inactive_contributors=0,

        number_of_good_contributors=2,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0],
        freerider_start_round=[1],
        malicious_noise_scale=[0],
        malicious_start_round=[1],
        malicious_attack_type=["byzantine"],
        freerider_attack_type=["delta_weight"],
        aggregation_rule=["FedAVG", "GRS_aggregation"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "p10_fedavg_vs_grs_freerider_cifar": FullPreset(
        fork=True,
        reward=int(0),
        # No reward — all users get the same reward each round anyway, so it's simplest to leave their GRS at 1.1 and 0.8.
        standard_buy_in=int(0.8e18),  # bad user's starting GRS
        min_buy_in=int(0.8e18),  # bad user's starting GRS
        max_buy_in=int(1.1e18),  # god brugers start-GRS
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=True,
        use_nobody_is_kicked=False,
        number_of_inactive_contributors=0,

        number_of_good_contributors=2,
        number_of_bad_contributors=0,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0],
        freerider_start_round=[1],
        malicious_noise_scale=[0],
        malicious_start_round=[1],
        malicious_attack_type=["byzantine"],
        freerider_attack_type=["delta_weight"],
        aggregation_rule=["FedAVG", "GRS_aggregation"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "p10_fedavg_vs_grs_malicious_mnist": FullPreset(
        fork=True,
        reward=int(0),
        # No reward — all users get the same reward each round anyway, so it's simplest to leave their GRS at 1.1 and 0.8.
        standard_buy_in=int(0.8e18),  # bad user's starting GRS
        min_buy_in=int(0.8e18),  # bad user's starting GRS
        max_buy_in=int(1.1e18),  # god brugers start-GRS
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=True,
        use_nobody_is_kicked=False,
        number_of_inactive_contributors=0,

        number_of_good_contributors=2,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=0,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0],
        freerider_start_round=[1],
        malicious_noise_scale=[0.01],
        malicious_start_round=[1],
        malicious_attack_type=["byzantine"],
        freerider_attack_type=["delta_weight"],
        aggregation_rule=["FedAVG", "GRS_aggregation"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "p10_fedavg_vs_grs_malicious_cifar": FullPreset(
        fork=True,
        reward=int(0),
        # No reward — all users get the same reward each round anyway, so it's simplest to leave their GRS at 1.1 and 0.8.
        standard_buy_in=int(0.8e18),  # bad user's starting GRS
        min_buy_in=int(0.8e18),  # bad user's starting GRS
        max_buy_in=int(1.1e18),  # god brugers start-GRS
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        force_merge_all=True,
        use_nobody_is_kicked=False,
        number_of_inactive_contributors=0,

        number_of_good_contributors=2,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=0,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        freerider_noise_scale=[0],
        freerider_start_round=[1],
        malicious_noise_scale=[0.01],
        malicious_start_round=[1],
        malicious_attack_type=["byzantine"],
        freerider_attack_type=["delta_weight"],
        aggregation_rule=["FedAVG", "GRS_aggregation"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,
        number_of_runs=10
    ),

    "comp_agg_mnist": FullPreset( # previously named: p10_MAIN_GRAPHS_comparing_aggregation_rules_mnist.
        # defaults
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=50,
        punish_factor=3,
        punish_factor_contrib=3,
        number_of_inactive_contributors=0,
        force_merge_all=True,      # NOT DEFAULT! experiment-specific
        use_nobody_is_kicked=True, # NOT DEFAULT! experiment-specific

        # always this
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,

        # best freerider
        freerider_start_round=[1],
        freerider_noise_scale=[0],
        freerider_attack_type=["delta_weight"],

        # best malicious
        malicious_start_round=None,
        malicious_noise_scale=[0.01],
        malicious_attack_type=["byzantine"],

        # dataset-specific
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,

        # experiment-specific
        aggregation_rule=["FedAVG", "positives_only", "plus_one_normalize", "binary_switch[positives_only, plus_one_normalize]", "partial_switch[retro, positives_only, plus_one_normalize]"],
        number_of_runs=10,
    ),


    "comp_agg_cifar": FullPreset(
        # defaults
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=0,
        punish_factor=3,
        punish_factor_contrib=3,
        number_of_inactive_contributors=0,
        force_merge_all=True,      # NOT DEFAULT! experiment-specific
        use_nobody_is_kicked=True, # NOT DEFAULT! experiment-specific

        # always this
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,

        # best freerider
        freerider_start_round=[1],
        freerider_noise_scale=[0.001],
        freerider_attack_type=["noise"],

        # best malicious
        malicious_start_round=None,
        malicious_noise_scale=[0.01],
        malicious_attack_type=["byzantine"],

        # dataset-specific
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,

        # experiment-specific
        aggregation_rule=["FedAVG", "positives_only", "plus_one_normalize", "binary_switch[positives_only, plus_one_normalize]", "partial_switch[retro, positives_only, plus_one_normalize]"],
        number_of_runs=10,
    ),


    "full_system_mnist": FullPreset(
        # defaults
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=0, # NOT DEFAULT! experiment-specific??
        punish_factor=3,
        punish_factor_contrib=3,
        number_of_inactive_contributors=0,
        force_merge_all=False,
        use_nobody_is_kicked=False,

        # always this
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,

        # best freerider
        freerider_start_round=[3],
        freerider_noise_scale=[0],
        freerider_attack_type=["delta_weight"],

        # best malicious
        malicious_start_round=None,
        malicious_noise_scale=[0.01],
        malicious_attack_type=["byzantine"],

        # dataset-specific
        number_of_good_contributors=4,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=10,
        epochs=1,
        batch_size=32,

        # experiment-specific
        aggregation_rule=["FedAVG", "positives_only", "plus_one_normalize", "binary_switch[positives_only, plus_one_normalize]",
                          "partial_switch[retro, positives_only, plus_one_normalize]", "GRS_aggregation"],
        number_of_runs=10,
    ),

    "full_system_cifar": FullPreset(
        # defaults
        fork=True,
        reward=int(1e18),
        standard_buy_in=int(1e18),
        min_buy_in=int(1e18),
        max_buy_in=int(1e18),
        first_round_fee=0,  # NOT DEFAULT! experiment-specific??
        punish_factor=3,
        punish_factor_contrib=3,
        number_of_inactive_contributors=0,
        force_merge_all=False,
        use_nobody_is_kicked=False,

        # always this
        use_outlier_detection=[True],
        contribution_score_strategy=["loss_only"],
        data_distribution=["random_split"],
        dirichlet_alpha=None,

        # best freerider
        freerider_start_round=[3],
        freerider_noise_scale=[0],
        freerider_attack_type=["delta_weight"],

        # best malicious
        malicious_start_round=None,
        malicious_noise_scale=[0.01],
        malicious_attack_type=["byzantine"],

        # dataset-specific
        number_of_good_contributors=6,
        number_of_bad_contributors=1,
        number_of_freerider_contributors=1,
        minimum_rounds=25,
        epochs=25,
        batch_size=128,

        # experiment-specific
        aggregation_rule=["FedAVG", "positives_only", "plus_one_normalize",
                          "binary_switch[positives_only, plus_one_normalize]",
                          "partial_switch[retro, positives_only, plus_one_normalize]", "GRS_aggregation"],
        number_of_runs=10,
    ),
}

# If you want to overwrite a value from the default preset, you need to create a FullPreset class inside PRESETS.
