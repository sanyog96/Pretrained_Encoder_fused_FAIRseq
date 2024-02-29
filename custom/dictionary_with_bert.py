from fairseq.tokenizer import tokenize_line
from fairseq.data import Dictionary

class DictionaryWithBert(Dictionary):
    def __init__(self, dic):
        super().__init__()
        for attr in ['unk_word', 'pad_word', 'eos_word',
                     'symbols', 'count', 'indices',
                     'bos_index', 'pad_index', 'eos_index', 'unk_index',
                     'nspecial', 'indices', ]:
            setattr(self, attr, getattr(dic, attr))

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                    consumer=None, append_eos=False, reverse_order=False):
        append_eos = False
        ids = super().encode_line(
            line, line_tokenizer, add_if_not_exist,
            consumer, append_eos, reverse_order)
        return ids
