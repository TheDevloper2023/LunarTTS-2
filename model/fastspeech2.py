import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .GST.HGST import HGST  # Hierarchical Global Style 
from .GST.tp_gst import TPSE
from transformers import RobertaModel, RobertaTokenizer

p_drop = 0.25


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, training=False):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)

        # GST
        self.HGST = None
        if model_config["gst"]["use_gst"]:
            self.HGST = HGST(preprocess_config, model_config)

            self.film_gamma = nn.Linear(model_config["transformer"]["encoder_hidden"],
                            model_config["transformer"]["encoder_hidden"])
            
            self.film_beta = nn.Linear(model_config["transformer"]["encoder_hidden"],
                                    model_config["transformer"]["encoder_hidden"])
            
            self.tpse_scale = nn.Parameter(torch.tensor(0.15))

            self.attn = nn.Sequential(
                nn.Linear(model_config["tpgst"]["BERT"]["bert_encoder_dim"],model_config["tpgst"]["BERT"]["bert_encoder_dim"]),
                nn.ReLU(),
                nn.Linear(model_config["tpgst"]["BERT"]["bert_encoder_dim"],1),
            )


        # Monke Patch 4 now
        self.a = nn.Linear(2048, 1280)
                        
        
        #TPSE
        self.bert_linear = TPSE(model_config)

        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        self.device = next(self.parameters()).device

        #roBERTa
        self.bert_train = model_config["tpgst"]["BERT"]["bert_train"]
        self.bert_encoder_dim = model_config["tpgst"]["BERT"]["bert_encoder_dim"]

        # Load pretrained BERT model
        bert_checkpoint = model_config["tpgst"]["BERT"]["bert_checkpoint_path"]
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(bert_checkpoint, local_files_only=False)
        self.roberta = RobertaModel.from_pretrained(bert_checkpoint, local_files_only=False).to(self.device) 


    # Train
    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        raw_texts,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        #RoBERTa
        bert_tokens = self.roberta_tokenizer(raw_texts, return_tensors="pt", padding=True)
        bert_tokens = bert_tokens.to("cuda:0") #TODO device
        with torch.no_grad():
            bert_outputs = self.roberta(**bert_tokens)



        # Combine the best of both (pooler - general intent; last hidden state - I forgot) to make it more expressive
        bert_cls = bert_outputs.pooler_output
        bert_hidden = bert_outputs.last_hidden_state

        weights = torch.softmax(self.attn(bert_hidden), dim=1)
        bert_attn = (weights * bert_hidden).sum(dim=1)
        
        bert_embed = torch.cat([bert_attn, bert_cls], dim=-1)

        bert_embed = bert_embed.unsqueeze(1)   # [B, 1, D]
        bert_embed_expanded = bert_embed.expand(-1, output.size(1), -1)

        # TPSE gen
        tpse_out = 0
        if self.bert_linear is not None:
            combined_embed = torch.cat([output, bert_embed_expanded], dim=-1)
            
            combined_embed = self.a(combined_embed)
            tpse_out = self.bert_linear(combined_embed)
            tpse_out = tpse_out.unsqueeze(1)          
        
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        
        # GST gen
        embedded_gst = self.HGST(mels, mel_lens)
        embedded_gst = F.dropout(embedded_gst, p_drop)
        #embedded_gst = embedded_gst.squeeze(1)

        

        gamma = self.film_gamma(embedded_gst)
        beta = self.film_beta(embedded_gst)
        
        gamma = gamma.expand(-1, output.size(1), -1)               # (B, T_enc, H_enc)
        beta  = beta.expand(-1, output.size(1), -1)

        # GST inject
        output = output * (gamma + 1) + beta

        
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            embedded_gst,
            tpse_out
        )
    

     # Validation / Eval / Synthesis
    def infer(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        raw_texts,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        #RoBERTa
        bert_tokens = self.roberta_tokenizer(raw_texts, return_tensors="pt", padding=True)
        bert_tokens = bert_tokens.to("cuda:0") #TODO device
        with torch.no_grad():
            bert_outputs = self.roberta(**bert_tokens)


        # Combine the best of both (pooler - general intent; last hidden state - I forgot) to make it more expressive
        bert_cls = bert_outputs.pooler_output
        bert_hidden = bert_outputs.last_hidden_state

        weights = torch.softmax(self.attn(bert_hidden), dim=1)
        bert_attn = (weights * bert_hidden).sum(dim=1)
        
        bert_embed = torch.cat([bert_attn, bert_cls], dim=-1)

        bert_embed = bert_embed.unsqueeze(1)   # [B, 1, D]
        bert_embed_expanded = bert_embed.expand(-1, output.size(1), -1)

        # TPSE gen
        tpse_out = 0
        if self.bert_linear is not None:
            combined_embed = torch.cat([output, bert_embed_expanded], dim=-1)
            
            combined_embed = self.a(combined_embed)
            tpse_out = self.bert_linear(combined_embed)
            embedded_gst = tpse_out.unsqueeze(1) if tpse_out.dim() == 2 else tpse_out         
        
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        
        gamma = self.film_gamma(embedded_gst)
        beta = self.film_beta(embedded_gst)
        
        gamma = gamma.expand(-1, output.size(1), -1)               # (B, T_enc, H_enc)
        beta  = beta.expand(-1, output.size(1), -1)

        # GST inject
        output = output * (gamma + 1) + beta

        
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            embedded_gst,
            tpse_out
        )
    