# FedCBA: Contribution-based Aggregation in Decentralized Federated Learning

Master's Thesis by:
- Caspar Emil Jensen, Lucas Lybek Højlund Pedersen and Rune Iversen Eberhardt
- Spring semester 2026 - Aalborg University
- Based on previous work from pre-specialisation thesis: https://github.com/SW9-10-DS-AAU/IHF-DFL

```
 _____   _____   ____     ____   ____       _    
|  ___| | ____| |  _ \   / ___| | __ )     / \   
| |_    |  _|   | | | | | |     |  _ \    / _ \  
|  _|   | |___  | |_| | | |___  | |_) |  / ___ \ 
|_|     |_____| |____/   \____| |____/  /_/   \_\                    
```

# Getting started

## 1. Clone & Git LFS

Install Git LFS (required before anything else):

**Linux:**
``` bash
sudo apt update 
sudo apt install git-lfs
```

**Windows:** Download and run the installer from https://git-lfs.com

**If you haven't cloned the repository yet:**

Then initialize Git LFS:
``` bash
git lfs install
git clone <repo-url>
```
**If you have already cloned the repo**, pull LFS files manually:
``` bash
git lfs install
git lfs pull
```

## 2. Ganache

- Download and install [Ganache](https://trufflesuite.com/ganache/)
- Create a new **workspace** (not Quickstart)
- Under **Server**, set the port to `7545`
- Under **Accounts & Keys**, set the number of accounts to `10`
- Under **Chain**, set the gas limit to a high value (e.g. `4000000000`), increase the default account balance to a higher amount (e.g. `10000000000`), and decrease the gas price (e.g. `200`)

## 3. Python Environment & Dependencies

Only tested with **Python 3.10**.

Create and activate a virtual environment:

**Linux:**
``` bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
``` bash
python -m venv .venv
.venv\Scripts\activate
```

Install the project and its dependencies:
``` bash
pip install -e ".[dev]"
```

**Strip notebook outputs before committing (run once per clone):**
``` bash
nbstripout --install
```

### Install a GPU backend — pick the one matching your hardware:

ROCM (AMD):
``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```
CUDA (NVIDIA):
``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## 4. Environment Variables

The project reads its configuration from `.env/<file>`. By default it uses `.env/.env.ganache`, which shall be configured for a standard local Ganache workspace (port 7545).


If another .env is preferred, run the program with the `ENV=<env_file_identifier>` prefix: 
```
ENV=ganache python ./experiment/experiments.py   # same as the default
ENV=sepolia python ./experiment/experiments.py   # example for a different network
```
Providing no ENV prefix and providing ENV=ganache is equivalent.


If you are connecting to a non-local blockchain (e.g. Sepolia), set these variables in your `.env` file:
```
RPC_URL="<RPC URL including port>"
PRIVATE_KEYS="<colon-separated private keys for your accounts>"
```
For Ganache (`fork=true`), `PRIVATE_KEYS` can be left empty.

## 5. Compile Smart Contracts

Build the ABI and bytecode files from the Solidity contracts:

**Linux:**
``` bash
python3 scripts/compile_contracts.py
```

**Windows:**
``` bash
python scripts/compile_contracts.py
```

## 6. Running an Experiment

Two modes are available:

### Sample
Quick demo of the full pipeline:
``` bash
python3 ./experiment/sample.py
```

### Experiment
Runs a systematic experiment across a parameter grid. 

To change the parameter sweep, edit `experiment/experiment_presets.py`. 

To run an experiment within the project, run the following command:
``` bash
python3 ./experiment/experiments.py
```
To change the dataset, edit `experiment/experiments.py`.
```
DATASETFAST = MNIST
DATASETSLOW = CIFAR.10
```

## 7. Solidity Testing

All forge commands must be run from the `foundry` directory.

Install Foundry (requires a Unix-like shell — WSL on Windows):
``` bash
curl -L https://foundry.paradigm.xyz | bash && source ~/.bashrc
foundryup
cd foundry && forge soldeer install
```

Build and test:
``` bash
cd foundry
forge build
forge test
```

## 8. Running Python Tests

Tests are run with pytest from the repo root:
``` bash
pytest
```

By default, tests marked as `slow` are skipped. These load full datasets (e.g. CIFAR-10) and can take significant time. To include them:
``` bash
pytest -m slow
```

To run only the slow tests:
``` bash
pytest -m slow --no-header -q
```

Solidity tests (forge) run automatically as part of the pytest suite on Linux/WSL.
On Windows they are skipped — run them manually in WSL:
``` bash
cd foundry && forge test
```

## 9. Test Coverage

Python:
``` bash
pytest --cov tests/
```

Solidity:
``` bash
forge coverage
```

Output to a file (optional):
``` bash
forge coverage --report lcov
```

## 10. Solidity Compiler (ARM)

If running on an ARM CPU, download a recompiled `solc` binary and place it at `~/.local/bin/solc`.
Binaries are available at: https://github.com/0xidm/solc-bin