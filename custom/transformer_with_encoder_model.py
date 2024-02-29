from collections import namedtuple
import torch
from transformers import BertModel, RobertaModel

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
)
from fairseq.models.fairseq_encoder import (
    EncoderOut,
)
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
)
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# EncoderOut = namedtuple('TransformerEncoderOut', [
#     'encoder_out',  # T x B x C
#     'encoder_padding_mask',  # B x T
#     'encoder_embedding',  # B x T x C
#     'encoder_states',  # List[T x B x C]
# ])

@register_model('transformer_with_pretrained_encoder')
class TransformerWithPretrainedEncoderModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        model = super().build_model(args, task)
        model.fine_tuning = args.fine_tuning
        return model

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerWithPretrainedEncoder(args, src_dict, embed_tokens)


    def train(self, mode=True):
        if self.fine_tuning:
            self.encoder.bert_model.train(mode)
            self.decoder.train(mode)
        else:
            self.encoder.bert_model.eval()
            self.decoder.train(mode)

    def eval(self):
        self.encoder.bert_model.eval()
        self.decoder.eval()


class TransformerWithPretrainedEncoder(FairseqEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.padding_idx = embed_tokens.padding_idx
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.fine_tuning = args.fine_tuning
        if(args.model_name == "BERT"):
            self.model = BertModel.from_pretrained(args.model,
                                                    output_hidden_states=True)
        elif(args.model_name == "RoBERTa"):
            self.model = RobertaModel.from_pretrained(args.model,
                                                    output_hidden_states=True)
        
        # peft_config = LoraConfig(
        #    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        # )
        # self.bert_model = get_peft_model(self.bert_model, peft_config)
        # self.bert_model.print_trainable_parameters()
        # print(sum(param.numel() for param in self.bert_model.parameters()))

        # for name, param in self.bert_model.named_parameters():
        #     if "10" in name:
        #         break
        #     param.requires_grad = False

    def forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x = None
        if not self.fine_tuning:
            with torch.no_grad():
                encoder_padding_mask = src_tokens.eq(self.padding_idx)
                attention_mask = src_tokens.ne(self.padding_idx).long()
                temp = self.model(input_ids=src_tokens,
                                    attention_mask=attention_mask)
                x = temp['last_hidden_state']
                layer_outputs = temp['hidden_states']
                # print(x)
                x = x.transpose(0, 1).detach()
                encoder_embedding = layer_outputs[0].detach()
                # x = layer_outputs[len(layer_outputs) - 2].transpose(0, 1).detach()
                encoder_states = None
                if return_all_hiddens:
                    encoder_states = [layer_outputs[i].transpose(0, 1).detach()
                                      for i in range(1, len(layer_outputs))]

        else:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            attention_mask = src_tokens.ne(self.padding_idx).long()
            temp = self.model(src_tokens,
                            attention_mask=attention_mask)
            x = temp['last_hidden_state']
            layer_outputs = temp['hidden_states']
            x = x.transpose(0, 1)
            encoder_embedding = layer_outputs[0]
            # x = layer_outputs[len(layer_outputs) - 2].transpose(0, 1).detach()
            encoder_states = None
            if return_all_hiddens:
                encoder_states = [layer_outputs[i].transpose(0, 1)
                                  for i in range(1, len(layer_outputs))]

        return EncoderOut(
            encoder_out = [x],  # T x B x C
            encoder_padding_mask = [encoder_padding_mask],  # B x T
            encoder_embedding = encoder_embedding,  # B x T x C
            encoder_states = encoder_states,  # List[T x B x C]
            src_tokens = src_tokens,
            src_lengths = src_lengths,
        )._asdict()

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

@register_model_architecture('transformer_with_pretrained_bert',
                             'transformer_with_pretrained_bert')
def transformer_with_pretrained_bert(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 768)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.activation_dropout = getattr(args, "activation_dropout", 0.3)
    args.dropout = getattr(args, "dropout", 0.3)
    # args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.2)
    base_architecture(args)

@register_model_architecture('transformer_with_pretrained_bert',
                             'transformer_with_pretrained_bert_768_3072')
def transformer_with_pretrained_bert(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.activation_dropout = getattr(args, "activation_dropout", 0.3)
    args.dropout = getattr(args, "dropout", 0.3)
    # args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.2)
    base_architecture(args)

@register_model_architecture('transformer_with_pretrained_bert',
                             'transformer_with_pretrained_bert_512')
def transformer_with_pretrained_bert_512(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.activation_dropout = getattr(args, "activation_dropout", 0.3)
    args.dropout = getattr(args, "dropout", 0.3)
    # args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.2)
    base_architecture(args)

@register_model_architecture('transformer_with_pretrained_bert',
                             'transformer_with_pretrained_bert_d5_emb512')
def transformer_with_pretrained_bert_512(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.activation_dropout = getattr(args, "activation_dropout", 0.3)
    args.dropout = getattr(args, "dropout", 0.3)
    # args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.2)
    base_architecture(args)

@register_model_architecture('transformer_with_pretrained_bert',
                             'transformer_with_pretrained_bert_large')
def transformer_with_pretrained_bert_large(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_architecture(args)
