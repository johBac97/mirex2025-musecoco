from fairseq.models.transformer import Linear
from fairseq.models import FairseqDecoder

import math, gc
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from .transformer_layer import LinearTransformerDecoderLayer


class LinearTransformerDecoder(FairseqDecoder):
    """
        Transformer decoder consisting of *args.decoder_layers* layers. Each layer
        is a :class:`TransformerDecoderLayer`.

        Args:
            args (argparse.Namespace): parsed command-line arguments
            dictionary (~fairseq.data.Dictionary): decoding dictionary
            embed_tokens (torch.nn.Embedding): output embedding
            no_encoder_attn (bool, optional): whether to attend to encoder outputs
                (default: False).
        """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                8192,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        # print(args.decoder_learned_pos)

        # self.embed_condition_pos = PositionalEmbedding(
        #     768,
        #     embed_dim,
        #     self.padding_idx,
        #     learned=True,
        # )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):  # args.decoder_normalize_before default False
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        self.gradient_checkpointing = getattr(self.args, 'gradient_checkpointing', False)
        if self.gradient_checkpointing:
            checkpointing_layers = getattr(self.args, 'gradient_checkpointing_layers', None)
            if checkpointing_layers is None:
                gradient_checkpointing_every_n_layer = getattr(self.args, 'gradient_checkpointing_every_n_layer', 1)
                checkpointing_layers = tuple(range(0, self.num_layers, gradient_checkpointing_every_n_layer))
            self.checkpointing_layers = checkpointing_layers

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return LinearTransformerDecoderLayer(args, no_encoder_attn)

    def forward(
            self,
            prev_output_tokens,
            sep_pos,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        '''
        According to fairseq_model.py:481, 
        prev_output_tokens = src_tokens 
        '''

        ''' Longshen's patch: when it's called first time, init the pos emb for input, exit '''
        # self.embed_condition_pos = PositionalEmbedding(
        #     768,
        #     1024,
        #     self.padding_idx,
        #     learned=False,
        # )
        # torch.save(self.state_dict(), "/data2/longshen/Checkpoints/musecoco/attribute2music1.pt")
        # exit(10)

        x, extra = self.extract_features(
            prev_output_tokens,
            sep_pos,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            sep_pos,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            sep_pos,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            sep_pos,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # Original Transformer's embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        # # Embed positions
        # embed_condition_pos = self.embed_positions
        # if self.embed_positions is not None:
        #     if not self.args.is_inference:
        #         # If in training,
        #         positions = []
        #         for i, cut_pos in enumerate(sep_pos):  # For a batch i with "cut_pos" condition tokens
        #             # # Original implementation
        #             # cur_word_positions = self.embed_positions(prev_output_tokens[i:i + 1, cut_pos:], incremental_state=None)
        #             # zeros_padding = torch.zeros([1, cut_pos, cur_word_positions.shape[2]]).to(cur_word_positions.device)
        #             # positions.append(torch.cat([zeros_padding, cur_word_positions], dim=1))
        #
        #             # Longshen: add positional embedding for input conditions
        #             cur_word_positions = self.embed_positions(prev_output_tokens[i:i + 1, cut_pos:],
        #                                                       incremental_state=None)
        #             input_positions = embed_condition_pos(prev_output_tokens[i:i + 1, :cut_pos],
        #                                                   incremental_state=None)
        #             positions.append(torch.cat([input_positions, cur_word_positions], dim=1))
        #
        #         positions = torch.cat(positions, dim=0)
        #     else:
        #         # If in inferencing,
        #         positions = []
        #         for i, cut_pos in enumerate(sep_pos):
        #             if prev_output_tokens.shape[1] > sep_pos[i]:  # sep_pos[i] means the absolute index
        #                 # If in 2 and later step of decoding, encode position for generated words.
        #                 # # Original code fill zeros as condition's position
        #                 # cur_word_positions = self.embed_positions(
        #                 #     prev_output_tokens[i:i + 1, cut_pos:],
        #                 #     incremental_state=None
        #                 # )
        #                 # zeros_padding = torch.zeros([1, cut_pos, cur_word_positions.shape[2]]).to(
        #                 #     cur_word_positions.device)
        #                 # positions.append(torch.cat([zeros_padding, cur_word_positions], dim=1))
        #
        #                 # Longshen: add condition position
        #                 cur_word_positions = self.embed_positions(prev_output_tokens[i:i + 1, cut_pos:],
        #                                                           incremental_state=None)
        #                 input_positions = embed_condition_pos(prev_output_tokens[i:i + 1, :cut_pos],
        #                                                       incremental_state=None)
        #                 positions.append(torch.cat([input_positions, cur_word_positions], dim=1))
        #             elif prev_output_tokens.shape[1] == sep_pos[
        #                 i]:  # For the first step of inference, just provide nothing
        #                 # else:
        #                 #     zeros_padding = torch.zeros([1, prev_output_tokens.shape[1], self.args.decoder_embed_dim]).to(
        #                 #         prev_output_tokens.device)
        #                 #     positions.append(zeros_padding)
        #                 input_positions = embed_condition_pos(
        #                     prev_output_tokens[i:i + 1, :cut_pos],
        #                     incremental_state=None
        #                 )
        #                 positions.append(input_positions)
        #             else:
        #                 input_positions = embed_condition_pos(
        #                     prev_output_tokens[i:i + 1, :cut_pos],
        #                     incremental_state=None
        #                 )
        #                 positions.append(input_positions)
        #
        #                 # print(prev_output_tokens.shape)
        #                 # print('sep pos', sep_pos[i])
        #                 # raise Exception('Check model input for inference!')
        #
        #         positions = torch.cat(positions, dim=0)
        # else:
        #     positions = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        gradient_checkpointing_every_n_layer = getattr(self.args, "gradient_checkpointing_every_n_layer", 1)

        for idx, layer in enumerate(self.layers):
            # if incremental_state is None and not full_context_alignment:
            #     self_attn_mask = self.buffered_future_mask(x)
            # else:
            #     self_attn_mask = None

            self_attn_mask = None  # Casual Linear Attention does not need this

            if (
                    getattr(self.args, "gradient_checkpointing", False) and self.training and
                    idx in self.checkpointing_layers
            ):
                x, layer_attn, _ = checkpoint(
                    layer,
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    None,
                    None,
                    self_attn_mask,
                    self_attn_padding_mask,
                    bool((idx == alignment_layer)),
                    bool((idx == alignment_layer)),
                )
            else:
                x, layer_attn, _ = layer(  # This line produce nan when fp16 training, when x.max()=1061
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features, **kwargs)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict
