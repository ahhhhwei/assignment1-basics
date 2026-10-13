import regex
from collections import Counter, defaultdict

# GPT-2 风格的预分词规则
PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# input_path：训练语料文件
# vocab_size：词表大小
# special_tokens：特殊 token 直接加入 vocab
def train_bpe(input_path, vocab_size: int, special_tokens: list[str]):
    # 1. 建立最初的词表：256 种 byte，世界上任何 UTF-8 文本都一定能编码，不会出现无法表示的字符
    vocab = {i: bytes([i]) for i in range(256)}
    # special token 也加入词表，不需要进行拆分
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges = [] # 记录 BPE 训练过程中到底按什么顺序进行了合并，供 tokenization 阶段使用
    # 2. 读取训练文本
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    # special token 不参与普通 BPE merge
    if special_tokens:
        special_pattern = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        text_parts = regex.split(special_pattern, text)
    else:
        text_parts = [text]
    # 3. 预分词，并统计每种 pre-token 出现多少次
    pretoken_counts = Counter()
    for part in text_parts:
        for match in PAT.finditer(part):
            token_bytes = match.group().encode("utf-8")
            pretoken_counts[token_bytes] += 1
    # 4. 每个 pre-token 最开始拆成单个 byte
    # b"low" -> [b"l", b"o", b"w"]
    words = []
    frequencies = []
    for word_bytes, freq in pretoken_counts.items():
        word = [bytes([b]) for b in word_bytes]
        words.append(word)
        frequencies.append(freq)
    # 5. 统计所有相邻 pair
    pair_counts = Counter()
    # pair -> 哪些 word 包含这个 pair
    pair_to_words = defaultdict(set)
    for word_id, word in enumerate(words):
        word_pairs = Counter(zip(word, word[1:]))
        for pair, count in word_pairs.items():
            pair_counts[pair] += (count * frequencies[word_id])
            pair_to_words[pair].add(word_id)
    # 6. 不断选择最高频 pair 进行 merge
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        # 频率最高的 pair
        # 如果频率一样，选择字典序更大的 pair
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        left, right = best_pair
        new_token = left + right
        # 记录 merge
        merges.append(best_pair)
        # 加入词表
        vocab[len(vocab)] = new_token
        # 只有包含 best_pair 的 word 才需要重新处理
        affected_words = list(pair_to_words[best_pair])
        for word_id in affected_words:
            old_word = words[word_id]
            freq = frequencies[word_id]
            # 先把旧 pair 的统计删除
            old_pairs = Counter(zip(old_word, old_word[1:]))
            for pair, count in old_pairs.items():
                pair_counts[pair] -= count * freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word_id)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]
            # 执行 merge
            new_word = []
            i = 0
            while i < len(old_word):
                if (i < len(old_word) - 1 and old_word[i] == left and old_word[i + 1] == right):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            words[word_id] = new_word
            # 加入新的 pair 统计
            new_pairs = Counter(zip(new_word, new_word[1:]))
            for pair, count in new_pairs.items():
                pair_counts[pair] += count * freq
                pair_to_words[pair].add(word_id)
    return vocab, merges

# 把 train_bpe() 训练得到的 vocab 和 merges，整理成真正编码时方便查询的数据结构
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # bytes -> token id（vocab 是 id → bytes，bytes_to_id 是 bytes → id）
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        # 每条 merge 规则的优先级
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        # special token -> token id
        self.special_token_to_id = {token: self.bytes_to_id[token.encode("utf-8")] for token in self.special_tokens}

    # 把一个 pre-token 的 bytes，按照训练好的 merges 规则，真正做 BPE 合并。
    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        # 最开始：拆成一个个 byte
        parts = [bytes([b]) for b in token_bytes]
        while len(parts) >= 2:
            best_pair = None
            best_rank = None
            # 找当前可以执行的、优先级最高的 merge
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            # 没有任何 pair 可以继续 merge
            if best_pair is None:
                break
            # 把所有 best_pair 合并
            new_parts = []
            i = 0
            while i < len(parts):
                if (i < len(parts) - 1 and parts[i] == best_pair[0] and parts[i + 1] == best_pair[1]):
                    new_parts.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    # 把普通文本 str 转成 token id 列表
    def _encode_normal_text(self, text: str) -> list[int]:
        ids = []
        for match in PAT.finditer(text):
            piece = match.group()
            piece_bytes = piece.encode("utf-8")
            bpe_tokens = self._bpe(piece_bytes)
            for token_bytes in bpe_tokens:
                ids.append(self.bytes_to_id[token_bytes])
        return ids

    # 把普通文本 str 转成 token id 列表。
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.special_tokens:
            return self._encode_normal_text(text)
        # 长的 special token 优先
        specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(regex.escape(token) for token in specials) + ")"
        parts = regex.split(pattern, text)
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_token_to_id:
                ids.append(self.special_token_to_id[part])
            else:
                ids.extend(self._encode_normal_text(part))
        return ids

    # 把 token id 列表还原回字符串
    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    # 对一批文本逐条编码，并且一个 token id 一个 token id 地产出，而不是一次性全部存进列表
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id