"""
config.py -- single source of truth. FROZEN before the final experiments.
Nothing here is tuned on test data. Change nothing between runs if you
want to reproduce the reported results.
"""
import os

# ---- experiment protocol (frozen) ----
SEEDS = [42, 123, 256, 789, 1024]
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 5
THRESHOLD_PERCENTILE = 95        # tau = 95th percentile of VALIDATION-benign error
DISPLAY_SEED = 42                # seed used for single-model figures (fixed in advance)

# ---- data ----
FEATURE_COLUMNS = [
    "Dur", "Proto", "SrcPkts", "DstPkts", "SrcBytes", "DstBytes",
    "SrcLoad", "DstLoad", "SrcLoss", "DstLoss", "sMeanPktSz", "dMeanPktSz",
    "sTtl", "dTtl", "sHops", "dHops", "sVid", "Offset", "Max", "Min",
]
LABEL_COLUMN = "Label"            # values: Benign / Malicious
ATTACK_TYPE_COLUMN = "Attack Type"
BENIGN_LABEL = "Benign"
MALICIOUS_LABEL = "Malicious"
PROTO_VALUES = ["arp", "icmp", "ipv6-icmp", "lldp", "llc", "sctp", "tcp", "udp"]

# ---- folders ----
DATA_DIR = "data"
RUN_DIR = "runs"
RESULTS_DIR = "results"
LOG_DIR = "logs"
for d in (DATA_DIR, RUN_DIR, RESULTS_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)
