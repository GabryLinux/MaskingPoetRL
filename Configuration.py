
class PATH_CONFIGURATION:
    @staticmethod
    def TOKENIZERS_PATH():
        return {
            "SYLLABLE": "tokenizers/syllable_tokenizer.json",
            "WORDPIECE": "tokenizers/wordpiece_tokenizer.json"
        }

    @staticmethod
    def MODELS_WEIGHTS_PATH():
        return {
            "SYLLABLE_TRANSFORMER": "./weights/syllable_transformer_weights.pt",
            "WORDPIECE_TRANSFORMER": "./weights/wordpiece_transformer_weights.pt",
            "SYLLABLE_CLASSIFIER": "./weights/syllable_classifier_weights.pt",
            "WORDPIECE_CLASSIFIER": "./weights/wordpiece_classifier_weights.pt",
            "SYLLABLE_MASKING_POLICY": "./weights/syllable_masking_policy_weights.pt",
            "WORDPIECE_MASKING_POLICY": "./weights/wordpiece_masking_policy_weights.pt",
            "SYLLABLE_MASKING_POLICY_BASELINE": "./weights/syllable_masking_policy_baseline_weights.pt",
            "WORDPIECE_MASKING_POLICY_BASELINE": "./weights/wordpiece_masking_policy_baseline_weights.pt",
            "SYLLABLE_MASKING_POLICY_REINFORCE": "./weights/syllable_masking_policy_reinforce_weights.pt",
            "WORDPIECE_MASKING_POLICY_REINFORCE": "./weights/wordpiece_masking_policy_reinforce_weights.pt",
            "SYLLABLE_MASKING_POLICY_REINFORCE_BASELINE": "./weights/syllable_masking_policy_reinforce_baseline_weights.pt",
            "WORDPIECE_MASKING_POLICY_REINFORCE_BASELINE": "./weights/wordpiece_masking_policy_reinforce_baseline_weights.pt",
        }

    @staticmethod
    def DATASETS_PATH():
        return {
            "POETRY": "../biblioteca_italiana/json",
            "NON-POETRY": "../biblioteca_italiana/json_parafrasi"
        }



class TRANSFORMER_CONFIGURATION:
    @staticmethod
    def PARAMETERS():
        return {
            "N_ATTENTION_LAYERS": 6,
            "N_HEADS_PER_LAYER": 8,
            "VECTOR_DIMENSION": 512,
            "MAX_SEQ_LEN": 256,
            "DROPOUT": 0.1
        }

    @staticmethod
    def TRAINING_PARAMETERS():
        return {
            "BATCH_SIZE": 32,
            "EPOCHS": 30,
            "LEARNING_RATE": 1e-3,
            "SEED": 67,
            "TRAIN_SPLIT_RATIO": 0.90  # 90% Training, 10% Testing
        }

class CLASSIFIER_CONFIGURATION:
    @staticmethod
    def PARAMETERS():
        return {
            "MAX_SEQ_LEN": TRANSFORMER_CONFIGURATION.PARAMETERS()["MAX_SEQ_LEN"],
            "EMBED_DIM": TRANSFORMER_CONFIGURATION.PARAMETERS()["VECTOR_DIMENSION"],
            "HIDDEN_DIM": TRANSFORMER_CONFIGURATION.PARAMETERS()["VECTOR_DIMENSION"],
            "WORDPIECE_LAYERS": [0, 1, 2, 3,4,5],
            "SYLLABLE_LAYERS": [0, 1, 2, 3,4,5],
            "DROPOUT": 0.3,
        }

    @staticmethod
    def TRAINING_PARAMETERS():
        return {
            "BATCH_SIZE": 64,
            "EPOCHS": 15,
            "LEARNING_RATE": 3e-4,
            "SEED": 67,
            "TRAIN_SPLIT_RATIO": 0.90  # 90% Training, 10% Testing
        }


class TOKENIZER_CONFIGURATION:
    @staticmethod
    def PARAMETERS():
        return {
            "WORDPIECE_VOCAB_SIZE": 15000,
        }

class RL_CONFIGURATION:
    @staticmethod
    def PARAMETERS():
        return {

        }
    @staticmethod
    def TRAINING_PARAMETERS():
        return {
            "EPOCHS": 1,
            "GAMMA": 0.90,
            "LR_WORDPIECE": 2e-4,
            "LR_SYLLABLE": 5e-5,
            "BATCH_SIZE": 32,
            "MAX_STEPS_PER_EPISODE": 12,
            "CUTOFF_WORDPIECE_THRESHOLD": 0.82,
            "CUTOFF_SYLLABLE_THRESHOLD": 0.72,
            "MAX_CHUNKS": 5000,
            "LOG_INTERVAL": 10,
            "TOP_K": 8,
            "SEED": 67,
        }