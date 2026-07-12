"""
stage2_inference.py
--------------------------------------------------------------------------------------------------
QUY TRÌNH 2 — LLM INFERENCE ONLY.

Việc thực hiện:
    1. Load lại file evidence JSON được tạo bởi stage1_retrieval.py.
    2. Với mỗi video, build lại LLM prompt (system + user) từ các thành phần evidence đã lưu
       (video_context_text, content_text, discourse_text, graph_text).
    3. Load model 1 lần, cho LLM trả lời lần lượt từng video (tuần tự).
    4. Ensemble LLM prediction với evidence lean/confidence đã lưu -> final_prediction.
    5. Lưu kết quả cuối cùng vào --checkpoint_path và kết thúc chương trình.

Resume an toàn: nếu --checkpoint_path đã tồn tại, các folder đã có trong đó sẽ được bỏ qua.

Usage:
    python stage2_inference.py \
        --evidence_path .../evidence_full.json \
        --checkpoint_path .../ablation_full.json \
        --model_name .../Qwen3-4B-Instruct-2507 \
        [--limit 50]   # chỉ xử lý tối đa 50 video (mới, chưa có trong checkpoint)
"""

import gc
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import common_utils as cu

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def count_tokens_qwen(tokenizer, messages: list) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))


def run_qwen_inference(tokenizer, model, messages: list, model_type: str = "instruct", max_new_tokens: int = 1024) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    if model_type == "thinking":
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    else:
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": 0.0}

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    output_ids = outputs[0][input_len:].tolist()
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if model_type == "thinking":
        thinking_token_id = 151668
        try:
            idx = len(output_ids) - output_ids[::-1].index(thinking_token_id)
            return tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
        except ValueError:
            return raw_output
    return raw_output


def load_evidence(evidence_path: Path) -> list:
    with open(evidence_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_result_checkpoint(checkpoint_path: Path) -> tuple:
    evaluation_results = []
    processed = set()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            processed = {r['folder_name'] for r in evaluation_results}
        except Exception:
            pass
    return evaluation_results, processed


def compute_final_prediction(evidence_mode, pred_label, entry):
    content_lean, content_conf   = entry.get("content_lean"), entry.get("content_conf")
    narrative_lean, narrative_conf = entry.get("narrative_lean"), entry.get("narrative_conf")
    graph_lean, graph_conf       = entry.get("graph_lean"), entry.get("graph_conf")

    if evidence_mode == "full":
        return cu.compute_ensemble_label_full(
            pred_label, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf
        )
    elif evidence_mode == "milvus":
        return cu.compute_ensemble_label_hybrid(
            pred_label, content_lean, content_conf, narrative_lean, narrative_conf
        )
    elif evidence_mode == "graph":
        return cu.compute_ensemble_label_two_signal(pred_label, graph_lean, graph_conf)
    else:
        return pred_label


def main(args: argparse.Namespace) -> None:
    evidence_path   = Path(args.evidence_path)
    checkpoint_path = Path(args.checkpoint_path)

    evidence_list = load_evidence(evidence_path)
    if not evidence_list:
        print(f"[WARN] No evidence entries found in {evidence_path}. Nothing to do.")
        return

    evidence_mode = args.evidence_mode or evidence_list[0].get("evidence_mode", "full")

    print(f"[INFO] Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    llm_model.eval()

    evaluation_results, processed_folders = load_result_checkpoint(checkpoint_path)

    # Lọc ra danh sách video chưa xử lý
    queue = [e for e in evidence_list if e['folder_name'] not in processed_folders]

    # Nếu có limit, chỉ lấy tối đa limit video đầu tiên
    if args.limit and args.limit > 0:
        original_count = len(queue)
        queue = queue[:args.limit]
        print(f"[INFO] Limit applied: processing only {len(queue)} / {original_count} remaining videos.")

    print(f"[INFO] {len(queue)} / {len(evidence_list)} videos left to run through the LLM (evidence_mode={evidence_mode}).")

    for current_idx, entry in enumerate(queue, 1):
        folder = entry['folder_name']
        print(f"[{current_idx}/{len(queue)}] {folder}", end=" ", flush=True)

        try:

            # # --- BẮT ĐẦU CHÈN ---
            # # Kiểm tra và rebuild graph_text nếu có graph_hits_record
            # graph_hits_record = entry.get("graph_hits_record", {})
            # if graph_hits_record and graph_hits_record.get("structural_matches"):
            #     # Rebuild graph_text với cách hiển thị mới (không cần tham số concept_label_name nữa)
            #     graph_text, graph_lean, graph_conf = cu.build_graph_text_from_hits(graph_hits_record)
            # else:
            #     # fallback nếu không có hits (dùng text cũ)
            #     graph_text = entry.get("graph_text", "")
            #     graph_lean = entry.get("graph_lean")
            #     graph_conf = entry.get("graph_conf")
            # # --- KẾT THÚC CHÈN ---

            # # Sau đó dùng graph_text đã có (có thể là mới hoặc cũ) để build prompt
            # llm_prompt = cu.build_llm_prompt(
            #     entry.get("video_context_text", ""),
            #     evidence_mode,
            #     entry.get("content_text"),
            #     entry.get("discourse_text"),
            #     graph_text,   # <-- thay vì entry.get("graph_text")
            # )

            llm_prompt = cu.build_llm_prompt(
                entry.get("video_context_text", ""),
                evidence_mode,
                entry.get("content_text"),
                entry.get("discourse_text"),
                entry.get("graph_text"),
            )
            system_prompt = cu.build_system_prompt(evidence_mode)

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": llm_prompt}]
            n_tokens = count_tokens_qwen(tokenizer, messages)

            llm_response = run_qwen_inference(tokenizer, llm_model, messages, model_type=args.model_type)

            verdict    = cu.extract_and_parse_json(llm_response)
            pred_label = verdict.get("predicted_label", -1)

            final_prediction = compute_final_prediction(evidence_mode, pred_label, entry)

            ground_truth = entry.get("ground_truth")
            print(f"| GT={ground_truth} final={final_prediction} | tokens={n_tokens:,}")

            evaluation_results.append({
                "folder_name":              folder,
                "evidence_mode":            evidence_mode,
                "ground_truth":             ground_truth,
                "prediction":               pred_label,
                "final_prediction":         final_prediction,
                "token_count":              n_tokens,
                "explanation":              verdict.get("explanation", ""),
                "improvement_suggestions":  verdict.get("improvement_suggestions", []),
                "raw_llm_output":           llm_response,
                "content_similarity_hits":  entry.get("content_hits_record", []),
                "narrative_evidence_hits":  entry.get("narrative_hits_record", []),
                "graph_evidence_hits":      entry.get("graph_hits_record", {}),
            })

        except Exception as e:
            print(f"| ERROR: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {len(evaluation_results)} predictions -> {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 — LLM inference from saved evidence")
    parser.add_argument("--evidence_path",   type=str, required=True,
                         help="JSON produced by stage1_retrieval.py.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                         help="Where to save final predictions.")
    parser.add_argument("--evidence_mode",   type=str, default=None,
                         help="Defaults to the evidence_mode stored in the evidence file.")
    parser.add_argument("--model_name",      type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_type",      type=str, default="instruct", choices=["instruct", "thinking"])
    parser.add_argument("--limit",           type=int, default=None,
                         help="Maximum number of videos to process (only for newly processed ones).")

    args = parser.parse_args()
    main(args)
