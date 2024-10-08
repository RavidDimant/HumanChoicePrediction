import wandb
YOUR_WANDB_USERNAME = "ravid-d"
project = "NLP2024_PROJECT_RavidDimant"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "seed": {"values": list(range(1, 6))},
        "bot_strategy_prob": {"values": [1]},
        "basic_nature": {"values": [18, 19, 20, 21, 22, 23, 26]},
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
# print("run this line to run your agent in a screen:")
# print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")

num_agents = 6
for i in range(num_agents):
    print(f"screen -dmS \"sweep_agent_{i}\" nohup wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")

