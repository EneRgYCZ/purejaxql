# Reinforcement Learning with Parallelized Q-Networks (PQN)

## Project Overview
This repository contains the implementation of transfer learning experiments in reinforcement learning using Parallelized Q-Networks (PQN) in the MinAtar environment. The goal is to evaluate the impact of transfer learning on the learning time of PQN, focusing on different configurations and their effects on training efficiency.

## Project Structure

```plaintext
.
├── docker
│   ├── build.sh             # Script to build the Docker image
│   ├── Dockerfile           # Docker configuration file
│   └── run.sh               # Script to run the Docker container
├── LICENSE                  # Project license
├── purejaxql
│   ├── config
│   │   ├── alg
│   │   │   ├── pqn_minatar_transfer.yaml  # Transfer learning configuration
│   │   │   └── pqn_minatar.yaml           # Baseline configuration
│   │   └── config.yaml      # General project configuration
│   ├── pqn_minatar.py       # Implementation of PQN in MinAtar
│   └── transfer_learning.py # Implementation of transfer learning experiments
├── README.md                # Project documentation
├── requirements
│   └── requirements.txt     # Python dependencies
├── results                  # Directory for storing results and plots
│   ├── Behavior-Policy-Asterix-From-Breakout.png
│   └── Target-Policy-Asterix-From-Breakout.png
├── run_multiple_sim_baseline.sh # Script to run baseline simulations
├── run_multiple_sim.sh      # Script to run multiple simulations
└── visualization
    ├── get_mean.py          # Script to compute mean rewards
    ├── make_plot.py         # Script to generate plots
```

## Usage (highly recommended with Docker)

Steps:

1. Ensure you have Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) properly installed.
2. (Optional) Set your WANDB key in the [Dockerfile](docker/Dockerfile).
3. Build with `bash docker/build.sh`.
5. Run a container: `bash docker/run.sh`.
6. Test a training script: `python purejaxql/pqn_minatar.py +alg=pqn_minatar`.

## Running the Experiments

### Baseline Simulations
Run baseline experiments using:
```bash
./run_multiple_sim_baseline.sh
```

### Transfer Learning Experiments
Run transfer learning experiments using:
```bash
./run_multiple_sim.sh
```

### Visualization
Use the scripts in the `visualization` directory to analyze and visualize results:

1. Compute mean rewards:
   ```bash
   python visualization/get_mean.py
   ```

2. Generate plots:
   ```bash
   python visualization/make_plot.py
   ```

## Reproducibility
The experiments are designed to be fully reproducible. Random seeds are explicitly set in the `config.yaml` file to ensure consistent results. All dependencies and configurations are contained within the Docker environment or specified in the requirements file.

## License
This project is licensed under the Apache 2.0 License. See the `LICENSE` file for more details.

## Related Projects

The following repositories are related to pure-GPU RL training:

- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [JaxMARL](https://github.com/FLAIROx/JaxMARL)
- [Jumanji](https://github.com/instadeepai/jumanji)
- [JAX-CORL](https://github.com/nissymori/JAX-CORL)
- [JaxIRL](https://github.com/FLAIROx/jaxirl)
- [Pgx](https://github.com/sotetsuk/pgx)
- [Mava](https://github.com/instadeepai/Mava)
- [XLand-MiniGrid](https://github.com/corl-team/xland-minigrid)
- [Craftax](https://github.com/MichaelTMatthews/Craftax/tree/main)

---

For further questions, please contact the project maintainer.
