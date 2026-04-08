import argparse
import json
import os
import sys
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(str(Path(__file__).parent / "RSTParser_EACL24" / "src"))
from data.tree import AttachTree
from parse.parse import parse_dataset


HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')


class RSTTreeParser:
    def __init__(self, args):
        self.base_model_name = args.base_model_name
        self.model_size = args.model_size
        self.parse_type = args.parse_type
        self.rel_type = args.rel_type
        self.corpus = args.corpus
        self.dataset_file = Path(args.dataset_file)

        self.model_type_list = self._get_model_type_list()

    def _get_model_type_list(self):
        if self.parse_type == "bottom_up":
            model_types = ["span"]
        elif self.parse_type == "top_down":
            model_types = ["top_down"]
        else:
            raise ValueError(f"Unknown parse_type: {self.parse_type}")

        if self.rel_type == "rel":
            model_types += ["nuc", "rel"]
        elif self.rel_type == "rel_with_nuc":
            model_types += ["nuc", "rel_with_nuc"]
        elif self.rel_type == "nuc_rel":
            model_types += ["nuc_rel"]
        else:
            raise ValueError(f"Unknown rel_type: {self.rel_type}")

        return model_types

    def _smart_tokenizer_and_embedding_resize(self, special_tokens_dict, tokenizer, model):
        num_new = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        if num_new > 0:
            input_embeddings_data = model.get_input_embeddings().weight.data
            output_embeddings_data = model.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings_data[:-num_new].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings_data[:-num_new].mean(dim=0, keepdim=True)
            input_embeddings_data[-num_new:] = input_embeddings_avg
            output_embeddings_data[-num_new:] = output_embeddings_avg

    def load_model(self):
        print(f"Loading base model: {self.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, token=HUGGING_FACE_TOKEN)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=HUGGING_FACE_TOKEN
        )
        self._smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tokenizer, base_model)

        # Build HF Hub IDs từ corpus + model_size
        hf_prefix = f"arumaekawa/{self.corpus}-{self.model_size}"
        adapter_hub_ids = {
            "span": f"{hf_prefix}-span",
            "top_down": f"{hf_prefix}-top_down",
            "nuc": f"{hf_prefix}-nuc",
            "rel": f"{hf_prefix}-rel",
            "nuc_rel": f"{hf_prefix}-nuc_rel",
            "rel_with_nuc": f"{hf_prefix}-rel_with_nuc",
        }

        print(f"Loading adapters from HF Hub: {self.model_type_list}")
        peft_model = None
        for model_type in self.model_type_list:
            hub_id = adapter_hub_ids[model_type]
            print(f" → {hub_id}")
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    base_model, hub_id,
                    adapter_name=model_type,
                    token=HUGGING_FACE_TOKEN
                )
            else:
                peft_model.load_adapter(hub_id, model_type, token=HUGGING_FACE_TOKEN)

        peft_model.eval()
        return peft_model, tokenizer

    # def _save_trees(self, output, save_dir, doc_id_to_video_dir):
    #     pred_dir = os.path.join(save_dir, "pred")
    #     os.makedirs(pred_dir, exist_ok=True)

    #     for doc_id, tree in zip(output["doc_id"], output["pred_tree"]):
    #         assert isinstance(tree, AttachTree)

    #         # Lưu vào save_path/pred/
    #         with open(os.path.join(pred_dir, f"{doc_id}.tree"), "w") as f:
    #             print(tree, file=f)

    #         # Lưu về folder video gốc nếu có
    #         if doc_id in doc_id_to_video_dir:
    #             video_dir = Path(doc_id_to_video_dir[doc_id])
    #             with open(video_dir / "rst_tree.tree", "w") as f:
    #                 print(tree, file=f)
    #             print(f"✅ {doc_id} → {pred_dir}/ + {video_dir}/rst_tree.tree")
    #         else:
    #             print(f"✅ {doc_id} → {pred_dir}/")

    def _save_trees(self, output, doc_id_to_video_dir):
        for doc_id, tree in zip(output["doc_id"], output["pred_tree"]):
            assert isinstance(tree, AttachTree)

            if doc_id in doc_id_to_video_dir:
                video_dir = Path(doc_id_to_video_dir[doc_id])
                with open(video_dir / "rst_tree.tree", "w") as f:
                    print(tree, file=f)
                print(f"{doc_id} → {video_dir}/rst_tree.tree")
            else:
                print(f"{doc_id}: no video_dir, pass")

    def run(self):
        print("=" * 80)
        print(f"parse_type : {self.parse_type}")
        print(f"rel_type : {self.rel_type}")
        print(f"corpus : {self.corpus}")
        print(f"adapters : {self.model_type_list}")
        print("=" * 80)

        # Load dataset
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"{len(dataset)} docs from {self.dataset_file}")

        doc_id_to_video_dir = {
            doc["doc_id"]: doc["video_dir"]
            for doc in dataset
            if "video_dir" in doc
        }

        # Load model
        model, tokenizer = self.load_model()

        # Parse
        print(" Inference ".center(80, "="))
        with torch.no_grad():
            output = parse_dataset(
                dataset, model, tokenizer,
                parse_type=self.parse_type,
                rel_type=self.rel_type,
                corpus=self.corpus,
            )

        # Save trees
        self._save_trees(output, doc_id_to_video_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RST Tree parsing cho video captions")

    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b", "70b"])
    parser.add_argument("--parse_type", type=str, default="bottom_up", choices=["bottom_up", "top_down"])
    parser.add_argument("--rel_type", type=str, default="rel_with_nuc", choices=["rel", "nuc_rel", "rel_with_nuc"])
    parser.add_argument("--corpus", type=str, default="rstdt", choices=["rstdt", "instrdt", "gum"])

    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path tới file JSON: list[{doc_id, edu_strings, video_dir?}]")

    args = parser.parse_args()

    rst_parser = RSTTreeParser(args)
    rst_parser.run()
        
