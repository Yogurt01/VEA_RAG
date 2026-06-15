import argparse
import json
import os
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(str(Path(__file__).parent / "RSTParser_EACL24" / "src"))
from data.tree import AttachTree
from parse.parse import parse_dataset

os.environ["HF_HOME"] = "/content/drive/MyDrive/KhoaLuan/models/huggingface_cache"
os.environ["TORCH_HOME"] = "/content/drive/MyDrive/KhoaLuan/models/torch_cache"
os.environ["HF_HUB_OFFLINE"] = "1"


class RSTTreeParser:
    def __init__(self, args):
        self.base_model_name = args.base_model_name
        self.adapter_base_path = args.adapter_base_path
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
        
        model_path = self.base_model_name

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
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
            local_files_only=True
        )
        self._smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tokenizer, base_model)

        adapter_base_path = self.adapter_base_path
        
        print(f"Loading adapters: {adapter_base_path}")
        peft_model = None
        
        for model_type in self.model_type_list:
            local_adapter_path = os.path.join(
                adapter_base_path, 
                f"{self.corpus}-{self.model_size}-{model_type}"
            )
            
            print(f" → Loading adapter [{model_type}] from: {local_adapter_path}")
            
            if not os.path.exists(local_adapter_path):
                raise FileNotFoundError(f"Không tìm thấy adapter tại: {local_adapter_path}")

            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    base_model, 
                    local_adapter_path,
                    adapter_name=model_type,
                    local_files_only=True
                )
            else:
                peft_model.load_adapter(
                    local_adapter_path, 
                    model_type, 
                    local_files_only=True
                )

        peft_model.eval()
        return peft_model, tokenizer

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
        start_time = time.time()
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
        print(f"\nAll processing steps completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RST Tree parsing cho video captions")

    parser.add_argument("--base_model_name", type=str, default="/content/drive/MyDrive/KhoaLuan/models/Llama-2-7b-hf")
    parser.add_argument('--adapter_base_path', type=str, default='/content/drive/MyDrive/KhoaLuan/models/adapters')
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b", "70b"])
    parser.add_argument("--parse_type", type=str, default="bottom_up", choices=["bottom_up", "top_down"])
    parser.add_argument("--rel_type", type=str, default="rel_with_nuc", choices=["rel", "nuc_rel", "rel_with_nuc"])
    parser.add_argument("--corpus", type=str, default="rstdt", choices=["rstdt", "instrdt", "gum"])
    parser.add_argument("--dataset_file", type=str, default="/content/drive/MyDrive/KhoaLuan/EnTube/edu_dataset.json", required=True, help="Path to file JSON: list[{doc_id, edu_strings, video_dir?}]")

    args = parser.parse_args()

    rst_parser = RSTTreeParser(args)
    rst_parser.run()
