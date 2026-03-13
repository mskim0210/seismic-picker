"""기본 설정값 (단일 소스).

모든 스크립트와 모듈에서 이 파일을 참조하여 기본 config를 사용.
학습된 모델 체크포인트에 config가 포함되어 있으면 체크포인트 config가 우선.
"""


def get_default_config():
    """기본 설정 dict 반환."""
    return {
        "model": {
            "name": "TPhaseNet",
            "in_channels": 3,
            "classes": 3,
            "filters_root": 8,
            "depth": 5,
            "kernel_size": 7,
            "stride": 2,
            "transformer_start_level": 3,
            "n_heads": 4,
            "ff_dim_factor": 4,
            "dropout": 0.1,
            "activation": "silu",
            "skip_attention": True,
            "lstm_hidden": 64,
        },
        "data": {
            "target_length": 6000,
            "sampling_rate": 100.0,
            "label_sigma": 20,
            "filter": {
                "enabled": True,
                "freq_min": 0.5,
                "freq_max": 45.0,
                "corners": 4,
            },
            "normalize": {
                "method": "std",
                "epsilon": 1e-8,
            },
        },
        "training": {
            "batch_size": 64,
            "num_workers": 4,
            "max_epochs": 100,
            "loss": "weighted_ce",
            "class_weights": [1.0, 30.0, 30.0],
            "optimizer": {
                "type": "adam",
                "lr": 1e-3,
                "weight_decay": 1e-5,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-6,
            },
            "gradient_clip": 1.0,
            "early_stopping": {"patience": 15},
            "mixed_precision": True,
            "checkpoint_dir": "./checkpoints",
        },
        "inference": {
            "device": "cuda",
            "peak_detection": {
                "min_height": 0.3,
                "min_distance": 100,
                "min_prominence": 0.1,
            },
            "sliding_window": {
                "window_size": 6000,
                "step": 3000,
            },
            "output_format": "json",
        },
    }
