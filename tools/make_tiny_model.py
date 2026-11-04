"""Build a tiny, randomly initialised Donut-architecture checkpoint for smoke tests.

The production model is ``naver-clova-ix/donut-base``.  This script creates a
miniature model with the *same architecture classes* (DonutSwin encoder +
MBart causal decoder inside a VisionEncoderDecoderModel, XLM-R style
tokenizer, DonutProcessor) so the full train -> evaluate -> inference pipeline
can be exercised on a CPU or in environments without Hugging Face Hub access.

Usage:
    python tools/make_tiny_model.py --dataset_path data_smoke --out tiny-donut
    python train.py --model_id tiny-donut --dataset_path data_smoke --image_size 320 320 ...
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers  # noqa: E402
from transformers import (  # noqa: E402
    DonutImageProcessor,
    DonutProcessor,
    DonutSwinConfig,
    DonutSwinModel,
    MBartConfig,
    MBartForCausalLM,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizerFast,
)

from p2n_format import ALL_SPECIAL_TOKENS, build_target_text  # noqa: E402

BASE_SPECIALS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def corpus(dataset_path, limit=20000):
    path = os.path.join(dataset_path, "metadata.jsonl")
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            for fmt in ("compact", "verbose"):
                yield build_target_text(json.loads(line)["ground_truth"], target_format=fmt)[1]
    # Make sure digits/punctuation are covered even for tiny corpora.
    yield "0123456789 -+.eE{}[]:,\"_ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ $\\^()/"


def build_tokenizer(dataset_path, vocab_size):
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=BASE_SPECIALS,
                                      unk_token="<unk>")
    tok.train_from_iterator(corpus(dataset_path), trainer=trainer)
    fast = XLMRobertaTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                   sep_token="</s>", cls_token="<s>")
    fast.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
    return fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True, help="Dataset used to train the tokenizer")
    ap.add_argument("--out", default="tiny-donut")
    ap.add_argument("--image_size", type=int, nargs=2, default=[320, 320])
    ap.add_argument("--vocab_size", type=int, default=1000)
    ap.add_argument("--embed_dim", type=int, default=48)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--decoder_layers", type=int, default=3)
    ap.add_argument("--max_position_embeddings", type=int, default=1536)
    args = ap.parse_args()

    tokenizer = build_tokenizer(args.dataset_path, args.vocab_size)
    h, w = args.image_size
    image_processor = DonutImageProcessor(size={"height": h, "width": w}, do_align_long_axis=False,
                                          do_thumbnail=True, do_pad=True)
    processor = DonutProcessor(image_processor=image_processor, tokenizer=tokenizer)

    enc_cfg = DonutSwinConfig(image_size=[h, w], patch_size=4, num_channels=3,
                              embed_dim=args.embed_dim, depths=[2, 2, 2, 2],
                              num_heads=[2, 4, 8, 8], window_size=5)
    dec_cfg = MBartConfig(vocab_size=len(tokenizer), d_model=args.d_model,
                          decoder_layers=args.decoder_layers, decoder_attention_heads=4,
                          decoder_ffn_dim=args.d_model * 4, encoder_layers=0,
                          max_position_embeddings=args.max_position_embeddings,
                          is_decoder=True, add_cross_attention=True, is_encoder_decoder=False,
                          scale_embedding=True, add_final_layer_norm=True,
                          pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
                          eos_token_id=tokenizer.eos_token_id)
    model = VisionEncoderDecoderModel(encoder=DonutSwinModel(enc_cfg),
                                      decoder=MBartForCausalLM(dec_cfg))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<s_p2n>")
    model.config.eos_token_id = tokenizer.eos_token_id

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    n = sum(p.numel() for p in model.parameters())
    print(f"Tiny Donut saved to {args.out}: {n/1e6:.2f}M params, vocab={len(tokenizer)}, image={h}x{w}")


if __name__ == "__main__":
    main()
