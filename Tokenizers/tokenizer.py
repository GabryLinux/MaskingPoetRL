import abc
import json
import os
import re
import unicodedata
from collections import Counter

import pyphen
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

from Configuration import PATH_CONFIGURATION

class BasePoetryTokenizer(abc.ABC):
    """
    Classe astratta base per i tokenizzatori di poesie.
    Definisce l'interfaccia standard che le classi figlie (Sillabe o WordPiece) 
    devono implementare per essere compatibili con le pipeline di dataset.
    """

    def __init__(self):
        """
        Inizializza le variabili base condivise da tutti i tokenizzatori.
        """
        self.special_tokens = ["[PAD]", "[UNK]", "[MASK]", "[VERSE]", "[STANZA]"]
        self.vocab = {}
        self.id_to_token = {}

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """
        Metodo astratto per convertire una stringa di testo in una lista di ID numerici.
        """
        pass

    @abc.abstractmethod
    def detokenize(self, token_ids: list[int], as_text: bool = True, control_tokens: bool = True) -> list[str] | str:
        """
        Metodo astratto per convertire una sequenza di ID nei token originali o in testo.
        """
        pass

    @abc.abstractmethod
    def save_tokenizer(self):
        """
        Metodo astratto per salvare lo stato del tokenizzatore su disco.
        """
        pass

    def get_special_tokens(self) -> list[str]:
        """
        Restituisce la lista dei token speciali di controllo configurati.
        """
        return self.special_tokens

    @abc.abstractmethod
    def train_from_corpus(self, poems: list[str], vocab_size: int = 30000):
        """
        Metodo astratto per addestrare il tokenizzatore su un corpus di poesie.
        """
        pass

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Rimuove segni diacritici, accenti e caratteri non-ASCII dal testo.
        """
        normalized = unicodedata.normalize('NFD', text)
        ascii_bytes = normalized.encode('ascii', 'ignore')
        return ascii_bytes.decode('utf-8')
    
    @staticmethod
    def parse_json_poems(json_path: str) -> list[str]:
        """
        Estrae le poesie da un singolo file JSON, inserendo i token di controllo.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        parsed_poems = []
        for poem in data:
            stanzas = poem.get("text", [])
            stanza_strings = []
            
            for stanza in stanzas:
                verses = [v.get("verse", "").strip() for v in stanza if v.get("verse", "").strip()]
                if verses:
                    stanza_str = " [VERSE] ".join(verses)
                    stanza_strings.append(stanza_str)
            
            poem_formatted = " [STANZA] ".join(stanza_strings)
            if poem_formatted:
                parsed_poems.append(poem_formatted)
                
        return parsed_poems

    @classmethod
    def get_all_poems_from_directory(cls, json_dir: str) -> list[str]:
        """
        Legge tutti i file JSON presenti nella cartella specificata e restituisce tutte le poesie.
        """
        all_poems = []
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"Directory non trovata: {json_dir}")

        for filename in sorted(os.listdir(json_dir)):
            if filename.endswith(".json"):
                file_path = os.path.join(json_dir, filename)
                poesie = cls.parse_json_poems(file_path)
                all_poems.extend(poesie)

        return all_poems


class SyllableTokenizer(BasePoetryTokenizer):
    """
    Implementazione del Tokenizzatore basato su Sillabe tramite Pyphen con supporto 
    alla ricomposizione esatta delle parole tramite prefissi sub-sillabici (##).
    """

    @staticmethod
    def from_config():
        return SyllableTokenizer(dict_path=PATH_CONFIGURATION.TOKENIZERS_PATH()["SYLLABLE"])

    def __init__(self, dict_path: str):
        super().__init__()
        self.dict_path = dict_path
        self.dic = pyphen.Pyphen(lang="it_IT")
        self._is_dirty = False
        self.counts = Counter()
        
        if os.path.exists(self.dict_path):
            with open(self.dict_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)
                
            if "counts" in tokenizer_data:
                self.counts = Counter(tokenizer_data["counts"])
                
            tokenizer_data = {k: v for k, v in tokenizer_data.items() if k != "counts"}
            self.tokenizer = Tokenizer.from_str(json.dumps(tokenizer_data, indent=4))
            self.vocab = self.tokenizer.get_vocab()
            
            for tok in self.special_tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
                    self._is_dirty = True
        else:
            self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
            self.tokenizer = Tokenizer(models.WordLevel(vocab=self.vocab, unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # type: ignore
            self.save_tokenizer()

        self._update_id_to_token_map()

    def _update_id_to_token_map(self):
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

    def save_tokenizer(self):
        self.tokenizer.model = models.WordLevel(vocab=self.vocab, unk_token="[UNK]") # type: ignore
        tokenizer_json_str = self.tokenizer.to_str()
        tokenizer_dict = json.loads(tokenizer_json_str)
        tokenizer_dict["counts"] = dict(self.counts)
        
        with open(self.dict_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_dict, f, ensure_ascii=False, indent=2)
            
        self._update_id_to_token_map()
        self._is_dirty = False

    def syllabify_word(self, word: str) -> list[str]:
        """
        Divide la parola in sillabe. La prima sillaba mantiene il formato originale, 
        mentre le successive ricevono il prefisso '##' per indicare la continuazione della parola.
        """
        word_clean = self.normalize_text(word.lower())
        syl_str = self.dic.inserted(word_clean)
        raw_syllables = syl_str.split('-') if syl_str else [word_clean]
        
        syllables = []
        for i, syl in enumerate(raw_syllables):
            # Aggiunge ## a tutte le sillabe tranne la prima della parola
            formatted_syl = syl if i == 0 else f"##{syl}"
            
            if formatted_syl not in self.vocab:
                self.vocab[formatted_syl] = len(self.vocab)
                self._is_dirty = True
            self.counts[formatted_syl] += 1
            syllables.append(formatted_syl)
                
        return syllables

    def syllabify_poem(self, poem_str: str, auto_save: bool = True) -> list[str]:
        tokens_and_words = re.findall(r'(\[[A-Z0-9_]+\]|\b\w+\b)', poem_str)
        syllables_and_tokens = []

        for item in tokens_and_words:
            if item.startswith("[") and item.endswith("]"):
                if item not in self.vocab:
                    self.vocab[item] = len(self.vocab)
                    self._is_dirty = True
                self.counts[item] += 1
                syllables_and_tokens.append(item)
            else:
                syllables_and_tokens.extend(self.syllabify_word(item))

        if auto_save and self._is_dirty:
            self.save_tokenizer()

        return syllables_and_tokens

    def train_from_corpus(self, poems: list[str], vocab_size: int = -1):
        print("[INFO] Estrazione delle sillabe dal corpus in corso...")
        for poem in poems:
            self.syllabify_poem(poem, auto_save=False)
            
        if vocab_size != -1:
            if vocab_size <= len(self.special_tokens):
                raise ValueError(f"vocab_size ({vocab_size}) deve essere maggiore del numero di token speciali ({len(self.special_tokens)}).")
                
            most_common = self.counts.most_common(vocab_size - len(self.special_tokens))
            new_vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            for tok, _ in most_common:
                if tok not in new_vocab:
                    new_vocab[tok] = len(new_vocab)
                    
            self.vocab = new_vocab
            self.tokenizer.model = models.WordLevel(vocab=self.vocab, unk_token="[UNK]") # type: ignore
            
        self._update_id_to_token_map()
        self._is_dirty = True
        print(f"[INFO] Vocabolario a sillabe pronto. Totale token: {len(self.vocab)}")

    def tokenize(self, text: str) -> list[int]:
        syllable_tokens = self.syllabify_poem(text, auto_save=False)
        unk_id = self.vocab.get("[UNK]", 1)
        return [self.vocab.get(tok, unk_id) for tok in syllable_tokens]

    def detokenize(self, token_ids: list[int], as_text: bool = True, control_tokens: bool = True) -> list[str] | str:
        """
        Ricostruisce il testo mantenendo attaccate le sillabe con prefisso '##' 
        e aggiungendo uno spazio solo prima delle nuove parole.
        Se control_tokens=True, preserva i token di controllo [VERSE] e [STANZA] nel testo.
        """
        unk_tok = "[UNK]"
        tokens = [self.id_to_token.get(idx, unk_tok) for idx in token_ids]

        if not as_text:
            return tokens

        text_elements = []
        for tok in tokens:
            if tok in ["[PAD]", "[UNK]"]:
                continue
            elif tok == "[VERSE]":
                if control_tokens:
                    text_elements.append(" [VERSE] ")
                else:
                    text_elements.append("\n")
            elif tok == "[STANZA]":
                if control_tokens:
                    text_elements.append(" [STANZA] ")
                else:
                    text_elements.append("\n\n")
            elif tok.startswith("##"):
                # Continuazione di parola: attacca senza spazio rimuovendo '##'
                text_elements.append(tok[2:])
            else:
                # Nuova parola: aggiunge uno spazio se non si trova all'inizio di una nuova riga o spazio
                if text_elements and not text_elements[-1].endswith("\n") and not text_elements[-1].endswith(" "):
                    text_elements.append(" " + tok)
                else:
                    text_elements.append(tok)

        text = "".join(text_elements)
        if control_tokens:
            text = re.sub(r' +', ' ', text).strip()
        return text

    def truncate_vocab(self, max_vocab_size: int, auto_save: bool = True):
        """
        Mantiene i token speciali e i (max_vocab_size - len(special_tokens)) token
        più frequenti presenti nel conteggio, riassegnando gli ID.
        """
        if max_vocab_size <= len(self.special_tokens):
            raise ValueError(
                f"max_vocab_size ({max_vocab_size}) deve essere maggiore del numero di token speciali ({len(self.special_tokens)})."
            )

        if len(self.vocab) <= max_vocab_size:
            print(f"[INFO] Il vocabolario ({len(self.vocab)}) è già inferiore o uguale a {max_vocab_size}. Nessun troncamento eseguito.")
            return

        # 1. Conserva i token speciali
        new_vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}

        # 2. Seleziona i token non speciali più frequenti
        # Esclude i token speciali dai conteggi da valutare
        filtered_counts = Counter({k: v for k, v in self.counts.items() if k not in new_vocab})
        num_to_keep = max_vocab_size - len(self.special_tokens)
        
        for tok, _ in filtered_counts.most_common(num_to_keep):
            new_vocab[tok] = len(new_vocab)

        # 3. Aggiorna lo stato interno del tokenizzatore
        self.vocab = new_vocab
        self.tokenizer.model = models.WordLevel(vocab=self.vocab, unk_token="[UNK]") # type: ignore
        self.counts = Counter({k: v for k, v in self.counts.items() if k in self.vocab})
        self._update_id_to_token_map()
        self._is_dirty = True

        if auto_save:
            self.save_tokenizer()

        print(f"[INFO] Vocabolario troncato con successo a {len(self.vocab)} token.")


class WordPieceTokenizer(BasePoetryTokenizer):
    """
    Implementazione del Tokenizzatore basato su sub-parole (WordPiece).
    """

    @staticmethod
    def from_config():
        """
        Inizializza il tokenizzatore WordPiece con i parametri definiti nella configurazione.
        """
        return WordPieceTokenizer(dict_path=PATH_CONFIGURATION.TOKENIZERS_PATH()["WORDPIECE"])

    def __init__(self, dict_path: str):
        super().__init__()
        self.dict_path = dict_path
        
        if os.path.exists(self.dict_path):
            self.tokenizer = Tokenizer.from_file(self.dict_path)
            self.vocab = self.tokenizer.get_vocab()
        else:
            self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # type: ignore
            self.tokenizer.decoder = decoders.WordPiece(prefix="##") # type: ignore
        
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}

    def train_from_corpus(self, poems: list[str], vocab_size: int = 30000):
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            continuing_subword_prefix="##"
        )
        
        print("[INFO] Addestramento WordPiece in corso...")
        normalized_poems = [self.normalize_text(p.lower()) for p in poems]
        self.tokenizer.train_from_iterator(normalized_poems, trainer)
        
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        self.save_tokenizer()
        print(f"[INFO] Addestramento completato. Dimensione vocabolario: {len(self.vocab)}")

    def save_tokenizer(self):
        self.tokenizer.save(self.dict_path)

    def tokenize(self, text: str) -> list[int]:
        normalized = self.normalize_text(text.lower())
        encoded = self.tokenizer.encode(normalized)
        return encoded.ids

    def detokenize(self, token_ids: list[int], as_text: bool = True, control_tokens: bool = True) -> list[str] | str:
        if not as_text:
            unk_tok = "[UNK]"
            return [self.id_to_token.get(idx, unk_tok) for idx in token_ids]

        decoded_str = self.tokenizer.decode(token_ids)
        if not control_tokens:
            decoded_str = decoded_str.replace("[VERSE]", "\n")
            decoded_str = decoded_str.replace("[STANZA]", "\n\n")
            decoded_str = re.sub(r' \n ', '\n', decoded_str)
        
        return decoded_str