"""
VideoRAG Inference Pipeline — Binary Classification (2-class)
with Milvus + Neo4j + google/gemma-4-E2B-it.

Labels:
    0 → Low Engagement  (Không thu hút)
    1 → High Engagement (Thu hút)

Usage:
    python inference_pipeline.py \
        --data_root       /path/to/All_Videos \
        --split_file      /path/to/dataset_splits.json \
        --checkpoint_path /path/to/output_checkpoint.json \
        [--model_name     /path/to/gemma-4-E2B-it] \
        [--count_tokens_only]

Environment variables required (loaded from .env or shell):
    MILVUS_CLUSTER_ENDPOINT
    MILVUS_TOKEN
    MILVUS_COLLECTION_NAME
    NEO4J_URI
    NEO4J_USERNAME
    NEO4J_PASSWORD
    NEO4J_DATABASE  (optional, default: neo4j)
"""

import os
import re
import gc
import time
import json
import argparse
import torch
from pathlib import Path
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText

load_dotenv()


# ==========================================
# 1. LABEL DEFINITIONS  (binary: 0 and 1)
# ==========================================

# Bộ nhãn nhị phân — không có nhãn 2
LABEL_DEFINITIONS = {
    0: "Low Engagement — The video fails to attract or retain viewers; narrative flow is weak or incoherent.",
    1: "High Engagement — The video effectively utilises multimodal elements and discourse structure to attract and retain viewers.",
}

# Tự động sinh block định nghĩa nhãn cho system prompt
_label_def_lines  = "\n".join(
    f"   - Label {lbl}: {desc}" for lbl, desc in sorted(LABEL_DEFINITIONS.items())
)
_valid_labels_str = " or ".join(str(k) for k in sorted(LABEL_DEFINITIONS.keys()))


# ==========================================
# 2. SYSTEM PROMPT  (full English)
# ==========================================

SYSTEM_PROMPT = f"""You are an Advanced Video Analysis and Evaluation System (VideoRAG Reasoning Engine).
Your task is to receive the structural, textual, and multimodal information of an input video,
cross-reference it with the similar contexts retrieved from the Vector Database and the
Knowledge Graph, and provide a final engagement prediction.

To reason accurately, you must master the following domain knowledge:

1. LABEL DEFINITIONS (Video Engagement Prediction — Binary):
{_label_def_lines}
   IMPORTANT: Only two labels exist in this dataset — 0 and 1.

2. MILVUS SEMANTIC MATCHING & QUERY SOURCES:
   - "Matched by: Video-level Mean Vector" indicates GLOBAL thematic similarity.
     The input video shares the overall topic/genre with the retrieved video.
   - "Matched by: Query Scene X" indicates LOCAL semantic similarity.
     A specific event, visual setting, or action in the input video mirrors a historical segment.

3. RST (Rhetorical Structure Theory) LOGIC CHAINS:
   - RST relations define how scenes connect logically
     (e.g., ELABORATION adds detail, CONTRAST highlights differences, CAUSE links reasons).
   - A structural match means the input video shares the EXACT SAME logical flow as a
     high-engagement (Label 1) or low-engagement (Label 0) historical video.

4. CONCEPT ALIGNMENT & STATISTICAL PRIORS:
   - When a scene belongs to a Concept node (e.g., Concept_G_C12), it aligns with a
     global semantic cluster mined from the entire training corpus.
   - The label_distribution field (e.g., {{"0": 3, "1": 8}}) shows how many
     videos of each engagement level contributed to that concept cluster.
   - Treat this distribution as a strong Bayesian prior when making your prediction.
"""


# ==========================================
# 3. CONFIG & CONSTANTS
# ==========================================

REQUIRED_DATA_KEYS = [
    'embeddings', 'scene_ids', 'metadata',
    'edge_index', 'edge_attr', 'rst_links', 'y',
]
OUTPUT_FIELDS = ["scene_uid", "video_id", "video_label", "caption"]

DEFAULT_MODEL_NAME = "google/gemma-4-E2B-it"

# Gemma-4-E2B context window — dùng để cảnh báo khi prompt quá dài
# Gemma 4 hỗ trợ 128k token, nhưng dùng 32k làm ngưỡng thực tế để tránh OOM
MAX_CONTEXT_TOKENS = 32768


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def generate_input_video_context(folder_name: str, data: dict, data_root: Path) -> str:
    """Build a text description of the input video from segments.json."""
    seg_path       = data_root / folder_name / "segments.json"
    captions_dict  = {}
    scene_ids_list = data['scene_ids']

    try:
        with open(seg_path, 'r', encoding='utf-8') as f:
            segments_content = json.load(f)

        if isinstance(segments_content, list):
            for idx, seg in enumerate(segments_content):
                if idx >= len(scene_ids_list):
                    break
                captions_dict[int(scene_ids_list[idx])] = seg.get(
                    'caption', "No text description available."
                )
        elif isinstance(segments_content, dict):
            for k, v in segments_content.items():
                captions_dict[int(k)] = (
                    v.get('caption', str(v)) if isinstance(v, dict) else str(v)
                )
    except Exception as e:
        print(f"  [WARNING] Cannot parse segments.json of {folder_name}: {e}")

    lines = [
        f"- Total Extracted Scenes: {len(scene_ids_list)}",
        "- Detailed Scene Textual Descriptions:",
    ]
    for scene_id in scene_ids_list[:30]:
        s_id_int = int(scene_id)
        caption  = captions_dict.get(s_id_int, "No caption text registered for this scene.")
        lines.append(f'  * Scene ID {s_id_int}: "{caption}"')

    rst_str = ", ".join(str(link) for link in data['rst_links'])
    lines.append(f"- Ground Rhetorical Structure Theory (RST) Links: [{rst_str}]")

    return "\n".join(lines)


def add_hits_to_pool(
    search_results,
    aggregated_hits: dict,
    query_scene_source: str,
) -> None:
    """Merge Milvus search results into a shared pool (dedup by max score)."""
    for hits in search_results:
        for hit in hits:
            entity = hit['entity']
            uid    = entity.get('scene_uid') or hit.get('id')
            score  = hit['distance']

            if uid not in aggregated_hits or score > aggregated_hits[uid]['score']:
                aggregated_hits[uid] = {
                    'score':            score,
                    'scene_uid':        uid,
                    'video_id':         entity.get('video_id',    'N/A'),
                    'video_label':      entity.get('video_label', 'N/A'),
                    'caption':          entity.get('caption',     'No caption available'),
                    'matched_by_query': query_scene_source,
                }


def reconstruct_multimodal_subgraph_with_logs(
    tx, video_id: str, data: dict, data_root: Path
) -> None:
    """Load test video subgraph into Neo4j (Video + Scene nodes + RST relationships)."""
    print(f"=== START RECONSTRUCTING MULTIMODAL GRAPH FOR VIDEO: {video_id} ===")

    seg_path      = data_root / video_id / "segments.json"
    segments_list = []

    if seg_path.exists():
        try:
            with open(seg_path, 'r', encoding='utf-8') as f:
                segments_list = json.load(f)
            print(f"--> Loaded {len(segments_list)} segments from JSON.")
        except Exception as e:
            print(f"--> [WARNING] Failed to load segments.json: {e}")
    else:
        print(f"--> [WARNING] segments.json not found at {seg_path}.")

    # Tạo node Video cho video test
    tx.run(
        "MERGE (v:Video {id: $video_id}) "
        "SET v.is_test = true, v.updated_at = timestamp()",
        video_id=video_id,
    )

    scene_query = """
    MATCH (v:Video {id: $video_id})
    MERGE (s:Scene {uid: $scene_uid})
    SET s.scene_id  = $scene_id,
        s.caption   = $caption,
        s.embedding = $embedding
    MERGE (v)-[:HAS_SCENE]->(s)
    """
    embeddings_list = data['embeddings'].tolist()
    scene_ids_list  = data['scene_ids']

    for idx, scene_id in enumerate(scene_ids_list):
        scene_id_int = int(scene_id)
        caption_val  = "No caption available"
        if idx < len(segments_list):
            seg_item    = segments_list[idx]
            caption_val = seg_item.get('caption', caption_val)

        tx.run(
            scene_query,
            video_id=video_id,
            scene_uid=f"{video_id}_scene_{scene_id_int}",
            scene_id=scene_id_int,
            caption=caption_val,
            embedding=embeddings_list[idx],
        )

    print(f"--> Uploaded {len(scene_ids_list)} Scene nodes.")

    # Tạo quan hệ RST giữa các Scene
    success_rel = 0
    for src, tgt, rel_type in data.get('rst_links', []):
        rel_type_upper = (
            str(rel_type).strip().upper().replace(" ", "_").replace("-", "_")
        )
        link_query = f"""
        MATCH (src:Scene {{uid: $src_uid}}), (tgt:Scene {{uid: $tgt_uid}})
        MERGE (src)-[:{rel_type_upper}]->(tgt)
        """
        tx.run(
            link_query,
            src_uid=f"{video_id}_scene_{int(src)}",
            tgt_uid=f"{video_id}_scene_{int(tgt)}",
        )
        success_rel += 1

    print(f"--> Connected {success_rel} RST relationships.")
    print("=== COMPLETED MULTIMODAL SUBGRAPH ===\n")


def get_global_sequence_structural_matched_subgraphs(tx, test_video_id: str) -> list:
    """
    Tìm các video trong Knowledge Graph có chuỗi RST tương tự video test.
    Trả về label_distribution (JSON string) thay vì num_videos_label_0/1 cũ.
    """
    query = """
    MATCH (v_test:Video {id: $test_video_id})

    MATCH p = (tsStart:Scene)-[
        :SIMILAR_TO|TEMPORAL|ELABORATION|CONTRAST|SPAN|ROOT|JOINT|CAUSE|
        TOPIC_COMMENT|EXPLANATION|EVALUATION|BACKGROUND|TOPIC_CHANGE|
        ATTRIBUTION|TEXTUAL_ORGANIZATION|COMPARISION|SUMMARY|SAME_UNIT|
        CONDITION|ENABLEMENT|MANNER_MEANS*2..4
    ]->(tsEnd:Scene)
    WHERE all(s IN nodes(p) WHERE (v_test)-[:HAS_SCENE]->(s))

    WITH v_test, p

    MATCH (v_cand:Video)
    WHERE v_cand.id <> $test_video_id
      AND v_cand.is_test IS NULL
      AND (v_cand.video_label IS NOT NULL
        OR v_cand.predicted_label IS NOT NULL
        OR v_cand.label IS NOT NULL)

    MATCH q = (csStart:Scene)-[
        :SIMILAR_TO|TEMPORAL|ELABORATION|CONTRAST|SPAN|ROOT|JOINT|CAUSE|
        TOPIC_COMMENT|EXPLANATION|EVALUATION|BACKGROUND|TOPIC_CHANGE|
        ATTRIBUTION|TEXTUAL_ORGANIZATION|COMPARISION|SUMMARY|SAME_UNIT|
        CONDITION|ENABLEMENT|MANNER_MEANS*2..4
    ]->(csEnd:Scene)
    WHERE all(s IN nodes(q) WHERE (v_cand)-[:HAS_SCENE]->(s))
      AND length(p) = length(q)
      AND all(i IN range(0, length(p)-1)
          WHERE type(relationships(p)[i]) = type(relationships(q)[i]))

    RETURN
        v_cand.id AS video_id,
        coalesce(v_cand.video_label, v_cand.predicted_label, v_cand.label) AS label,
        max(length(p) + 1) AS max_nodes_matched,
        count(distinct q)  AS total_matched_sequences,
        collect({
            length:         length(p) + 1,
            sequence_ids:   [s IN nodes(q) | s.scene_id],
            relation_chain: [r IN relationships(q) | type(r)],
            scene_captions: [s IN nodes(q) | coalesce(s.caption, "No caption")],
            scene_concepts: [s IN nodes(q) | [
                (s)-[:BELONGS_TO]->(c:Concept) | {
                    concept_id:         c.id,
                    label_distribution: c.label_distribution
                }
            ][0]]
        }) AS match_details
    ORDER BY max_nodes_matched DESC, total_matched_sequences DESC
    LIMIT 5
    """
    return list(tx.run(query, test_video_id=test_video_id))


def format_concept_distribution(label_distribution_json: str) -> str:
    """
    Chuyển label_distribution JSON string thành chuỗi hiển thị động.
    Input  : '{"0": 3, "1": 8}'
    Output : 'L0=3, L1=8'
    """
    if not label_distribution_json:
        return "No distribution data"
    try:
        dist = json.loads(label_distribution_json)
        return ", ".join(
            f"L{k}={v}" for k, v in sorted(dist.items(), key=lambda x: int(x[0]))
        )
    except Exception:
        return label_distribution_json  # fallback: trả về raw string


def extract_and_parse_json(raw_text: str) -> dict:
    """
    Parse JSON từ LLM output.
    Với bộ nhãn nhị phân {0, 1}, validate giá trị trả về thuộc tập hợp hợp lệ.
    """
    valid_labels = set(LABEL_DEFINITIONS.keys())   # {0, 1}

    if not raw_text:
        return {"predicted_label": -1, "explanation": "Empty LLM response"}

    # Xoá markdown code fence nếu LLM tự thêm vào
    clean = raw_text.strip()
    clean = re.sub(r'^```json\s*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'^```\s*',     '', clean)
    clean = re.sub(r'\s*```$',     '', clean)
    clean = clean.strip()

    # Thử parse JSON trực tiếp
    try:
        match  = re.search(r'\{.*\}', clean, re.DOTALL)
        target = match.group(0) if match else clean
        parsed = json.loads(target)
        pred   = parsed.get("predicted_label")
        if pred is not None:
            pred_int = int(str(pred).strip())
            # Nếu LLM trả về 2 (nhãn không hợp lệ trong bộ này),
            # ánh xạ về nhãn hợp lệ gần nhất theo ngưỡng giữa (0 vs 1)
            if pred_int not in valid_labels:
                print(f"  [LABEL REMAP] LLM returned invalid label {pred_int}, remapping...")
                pred_int = min(valid_labels, key=lambda x: abs(x - pred_int))
                print(f"  [LABEL REMAP] Remapped to: {pred_int}")
            parsed["predicted_label"] = pred_int
            return parsed
    except Exception:
        pass

    # Fallback: regex bắt số nguyên sau predicted_label
    broad = re.search(
        r'(predicted[-_ ]label|label|prediction)\s*["\']?\s*[:=]\s*["\']?\s*(\d+)',
        clean, re.IGNORECASE,
    )
    if broad:
        pred_int = int(broad.group(2))
        if pred_int not in valid_labels:
            pred_int = min(valid_labels, key=lambda x: abs(x - pred_int))
        return {"predicted_label": pred_int, "explanation": "Broad Regex"}

    # Fallback cuối: quét văn bản thô
    raw_scan = re.search(
        r'predicted[-_ ]label\s+(?:is|should be)\s+(\d+)', clean, re.IGNORECASE
    )
    if raw_scan:
        pred_int = int(raw_scan.group(1))
        if pred_int not in valid_labels:
            pred_int = min(valid_labels, key=lambda x: abs(x - pred_int))
        return {"predicted_label": pred_int, "explanation": "Raw text scan"}

    return {"predicted_label": -1, "explanation": "Parsing failed"}


def extract_gemma_response(parsed_response) -> str:
    """
    Chuẩn hoá output của processor.parse_response() thành plain string.
    Gemma 4 trả về dict hoặc string tuỳ phiên bản transformers.
    """
    if parsed_response is None:
        return ''
    if isinstance(parsed_response, dict):
        return (parsed_response.get('content') or parsed_response.get('text') or '').strip()
    return str(parsed_response).strip()


def count_tokens_gemma(processor, messages: list) -> int:
    """
    Đếm số token của danh sách messages sau khi áp dụng chat template Gemma 4.
    Dùng để ước lượng độ dài prompt TRƯỚC khi gọi LLM.
    """
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # processor.tokenizer cho phép encode text thành token ids
    return len(processor.tokenizer.encode(text))


def build_llm_prompt(
    video_context_text: str,
    milvus_context_text: str,
    neo4j_context_text: str,
) -> str:
    """Tạo user prompt đầy đủ cho LLM inference."""
    return f"""=== 1. CURRENT INPUT VIDEO STRUCTURE & CONTENT ===
{video_context_text}

=== 2. CORE SEMANTIC SIMILAR CONTEXT (From Milvus Vector DB) ===
{milvus_context_text}

=== 3. RELEVANT LOGICAL STRUCTURAL CONTEXT (From Neo4j Knowledge Graph) ===
{neo4j_context_text}

=== 4. IN-CONTEXT REASONING EXAMPLE (FEW-SHOT) ===
The following example illustrates how to reason over structural matches and multi-level semantic alignment.
---
[Example Scenario]:
- Milvus Top 1 Match (Video-level Mean Vector) -> Video 'V_High' (Label 1): overall topic is highly engaging.
- Milvus Top 2 Match (Query Scene 2) -> Video 'V_Low' (Label 0): this specific scene setup historically caused viewer drop-offs.
- Neo4j Structural Match: current RST chain is identical to Video 'V_High' (Label 1).
- Concept Alignment: Scene 3 -> Concept_G_C12 (Related Videos: L0=2, L1=8) — strong Label 1 prior.

[Example Expected Output]:
{{
  "predicted_label": "1",
  "explanation": "Although Scene 2 shows a local semantic risk by matching a low-engagement video 'V_Low' (Label 0), the global topic alignment with 'V_High' (Label 1) and the Concept cluster prior (L0=2, L1=8) strongly favour Label 1. The RST structural symmetry with 'V_High' further confirms a high-engagement prediction.",
  "improvement_suggestions": [
    "Rework Scene 2 visually to avoid the low-engagement pattern seen in 'V_Low'.",
    "Preserve the overall RST chain, as it replicates a proven high-engagement discourse structure."
  ]
}}
---

=== REASONING GUIDELINES ===
1. Distinguish global thematic matches (Video-level Mean Vector) from local scene matches (Query Scene X) in the Milvus results.
2. Synthesise the Milvus label distribution (Top-5 counts), the Neo4j Concept label_distribution priors (L0/L1 counts), and the RST structural chains into a single coherent judgment.
3. In the 'explanation', explicitly identify whether the decision is driven by global thematic alignment, scene-level semantic matches, or structural logic symmetry.
4. This is a BINARY task — only Label 0 (Low Engagement) and Label 1 (High Engagement) are valid.

=== OUTPUT COMPLIANCE ===
Return your judgment as a strict JSON object with no additional text outside the braces:
{{
  "predicted_label": "{_valid_labels_str}",
  "explanation": "Deep multimodal analysis connecting global/local semantic matches, RST chains, and Concept label_distribution to the final label.",
  "improvement_suggestions": [
    "Actionable suggestion 1 targeting identified structural deficiencies or semantic mismatches.",
    "Actionable suggestion 2 to optimise video production or narrative flow."
  ]
}}
"""


# ==========================================
# 5. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:

    data_root             = Path(args.data_root)
    split_dataset_file    = Path(args.split_file)
    batch_checkpoint_path = Path(args.checkpoint_path)

    # --- Đọc biến môi trường Milvus (KHÔNG qua args) ---
    milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
    milvus_token    = os.getenv("MILVUS_TOKEN")
    collection_name = os.getenv("MILVUS_COLLECTION_NAME")

    # --- Đọc biến môi trường Neo4j (KHÔNG qua args) ---
    neo4j_uri      = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([milvus_endpoint, milvus_token, collection_name]):
        raise ValueError(
            "Missing Milvus environment variables. "
            "Please set MILVUS_CLUSTER_ENDPOINT, MILVUS_TOKEN, MILVUS_COLLECTION_NAME."
        )
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError(
            "Missing Neo4j environment variables. "
            "Please set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )

    # ------------------------------------------------------------------
    # Load processor và model google/gemma-4-E2B-it
    # ------------------------------------------------------------------
    print(f"[INFO] Loading processor & model from: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,   # Gemma 4 khuyến nghị bfloat16
        device_map="auto",
    )
    model.eval()
    print("[INFO] Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Khởi tạo kết nối Milvus
    # ------------------------------------------------------------------
    milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)
    print("[INFO] Milvus connection established.")

    # ------------------------------------------------------------------
    # Khởi tạo kết nối Neo4j
    # ------------------------------------------------------------------
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    try:
        driver.verify_connectivity()
        print("[INFO] Neo4j connection established.\n")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to Neo4j: {e}")

    # ------------------------------------------------------------------
    # Validation dữ liệu: kiểm tra file tồn tại và keys đầy đủ
    # ------------------------------------------------------------------
    with open(split_dataset_file, "r") as f:
        splits = json.load(f)
    test_folders = splits["test"]

    valid_folders:      list = []
    valid_folders_data: dict = {}
    invalid_folders:    list = []

    for folder_name in test_folders:
        folder_path = data_root / folder_name
        emb_path    = folder_path / "scene_embeddings.pt"
        seg_path    = folder_path / "segments.json"

        missing = [
            name for name, p in [
                ("scene_embeddings.pt", emb_path),
                ("segments.json",       seg_path),
            ] if not p.exists()
        ]
        if missing:
            invalid_folders.append((folder_name, f"Missing file(s): {', '.join(missing)}"))
            continue

        try:
            data         = torch.load(emb_path, map_location='cpu')
            missing_keys = [k for k in REQUIRED_DATA_KEYS if k not in data]
            if missing_keys:
                invalid_folders.append((folder_name, f"Missing keys in .pt: {missing_keys}"))
            else:
                valid_folders.append(folder_name)
                valid_folders_data[folder_name] = data
        except Exception as e:
            invalid_folders.append((folder_name, f"Corrupted .pt file: {e}"))

    print(f"Validation: {len(valid_folders)} valid, {len(invalid_folders)} invalid.")
    if invalid_folders:
        for f, reason in invalid_folders:
            print(f"  [INVALID] {f}: {reason}")

    # ------------------------------------------------------------------
    # TOKEN COUNTING MODE  (--count_tokens_only)
    #
    # Chạy chế độ này TRƯỚC lần inference đầu tiên để:
    #   1. Biết số token tối đa / trung bình / tối thiểu của từng video
    #   2. Quyết định MAX_CONTEXT_TOKENS phù hợp cho Gemma 4
    #   3. Phát hiện video nào có prompt quá dài cần xử lý đặc biệt
    #
    # Lưu ý: chế độ này KHÔNG query Milvus / Neo4j, chỉ dùng dummy context
    # để ước lượng. Token thực tế khi có đủ context sẽ CAO HƠN con số này.
    # ------------------------------------------------------------------
    if args.count_tokens_only:
        print("\n" + "="*70)
        print(" TOKEN COUNTING MODE — no LLM inference will be run")
        print(" NOTE: counts below are LOWER BOUNDS (Milvus + Neo4j context excluded)")
        print("="*70)

        token_counts: list = []

        for folder in valid_folders:
            sample_data = valid_folders_data[folder]

            video_context_text = generate_input_video_context(folder, sample_data, data_root)

            # Dùng placeholder để ước lượng — không gọi Milvus / Neo4j
            dummy_milvus = "[Milvus context placeholder — not fetched in token-count mode]"
            dummy_neo4j  = "[Neo4j context placeholder — not fetched in token-count mode]"
            llm_prompt   = build_llm_prompt(video_context_text, dummy_milvus, dummy_neo4j)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": llm_prompt},
            ]
            n_tokens = count_tokens_gemma(processor, messages)
            token_counts.append({"folder": folder, "token_count": n_tokens})
            print(f"  {folder}: {n_tokens:>7,} tokens")

        # Thống kê để quyết định context window
        all_counts = [r["token_count"] for r in token_counts]
        print("\n" + "="*70)
        print(f" Total videos counted          : {len(all_counts)}")
        print(f" Min tokens (shortest prompt)  : {min(all_counts):>7,}")
        print(f" Max tokens (longest prompt)   : {max(all_counts):>7,}")
        print(f" Mean tokens                   : {sum(all_counts)/len(all_counts):>10,.1f}")
        print(f" Videos exceeding {MAX_CONTEXT_TOKENS:,} tokens: "
              f"{sum(1 for c in all_counts if c > MAX_CONTEXT_TOKENS)}")
        print("="*70)

        # Lưu kết quả ra JSON để tham khảo sau
        token_count_path = batch_checkpoint_path.parent / "token_counts_gemma_binary.json"
        with open(token_count_path, 'w', encoding='utf-8') as f:
            json.dump(token_counts, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Token count results saved to: {token_count_path}")

        driver.close()
        return  # Dừng tại đây, không chạy inference

    # ------------------------------------------------------------------
    # Khôi phục checkpoint nếu pipeline bị gián đoạn
    # ------------------------------------------------------------------
    evaluation_results: list = []
    processed_folders:  set  = set()

    if batch_checkpoint_path.exists():
        try:
            with open(batch_checkpoint_path, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            processed_folders = {res['folder_name'] for res in evaluation_results}
            print(f"[INFO] Resuming from checkpoint. Already processed: {len(processed_folders)} videos.")
        except Exception as e:
            print(f"[WARNING] Corrupted checkpoint, starting fresh: {e}")

    final_queue = [f for f in valid_folders if f not in processed_folders]
    print(f"[START] Will process {len(final_queue)} videos.\n")

    success_count = 0

    for current_idx, folder in enumerate(final_queue, 1):
        print(f"\n{'='*70}")
        print(f" $$ PROCESSING VIDEO [{current_idx}/{len(final_queue)}]: {folder}")
        print("="*70)

        test_video_id   = folder
        sample_data     = valid_folders_data[folder]
        aggregated_hits: dict = {}   # reset pool Milvus cho từng video
        milvus_context_text   = ""
        neo4j_context_text    = ""

        try:
            # --------------------------------------------------------------
            # STEP A: Build input video context text
            # --------------------------------------------------------------
            video_context_text = generate_input_video_context(folder, sample_data, data_root)

            # --------------------------------------------------------------
            # STEP B: Query Milvus — local (per-scene) + global (mean vector)
            # --------------------------------------------------------------
            scene_embeddings = sample_data['embeddings']
            num_scenes       = scene_embeddings.shape[0]

            # Tìm kiếm theo từng scene riêng lẻ (local semantic match)
            for i in range(num_scenes):
                res_scene = milvus_client.search(
                    collection_name=collection_name,
                    data=[scene_embeddings[i].tolist()],
                    limit=5,
                    output_fields=OUTPUT_FIELDS,
                )
                add_hits_to_pool(
                    res_scene, aggregated_hits,
                    query_scene_source=f"Query Scene {sample_data['scene_ids'][i]}",
                )

            # Tìm kiếm theo vector trung bình toàn video (global thematic match)
            res_mean = milvus_client.search(
                collection_name=collection_name,
                data=[scene_embeddings.mean(dim=0).tolist()],
                limit=5,
                output_fields=OUTPUT_FIELDS,
            )
            add_hits_to_pool(res_mean, aggregated_hits, query_scene_source="Video-level Mean Vector")

            top_5_hits = sorted(
                aggregated_hits.values(), key=lambda x: x['score'], reverse=True
            )[:5]

            # Đếm động số lượng hit theo từng nhãn
            label_hit_counts: dict = {}
            for hit in top_5_hits:
                lbl = hit['video_label']
                if isinstance(lbl, int):
                    label_hit_counts[lbl] = label_hit_counts.get(lbl, 0) + 1

            label_hit_summary = ", ".join(
                f"Label {lbl}: {cnt}" for lbl, cnt in sorted(label_hit_counts.items())
            )

            detailed_matches_text = ""
            for idx, hit in enumerate(top_5_hits, 1):
                s_uid = hit['scene_uid']
                retrieved_scene_idx = s_uid.split('_')[-1] if '_' in str(s_uid) else "Unknown"
                detailed_matches_text += (
                    f"- Top {idx} Match (Cosine Score: {hit['score']:.4f}):\n"
                    f"  Matched by: {hit['matched_by_query']}\n"
                    f"  Target Location: Video '{hit['video_id']}' at Scene '{retrieved_scene_idx}'\n"
                    f"  Target Video Ground-Truth Label: {hit['video_label']}\n"
                    f"  Scene Caption: {hit['caption']}\n\n"
                )

            milvus_context_text = (
                "=== MILVUS VECTOR RETRIEVAL SUMMARY ===\n"
                f"Top-5 most similar retrieved segments — {label_hit_summary}.\n\n"
                "=== DETAILED RETRIEVED SEGMENTS ===\n"
                + detailed_matches_text
            )

            # --------------------------------------------------------------
            # STEP C: Load test subgraph vào Neo4j + truy vấn structural match
            # --------------------------------------------------------------
            with driver.session(database=neo4j_database) as session:
                session.execute_write(
                    reconstruct_multimodal_subgraph_with_logs,
                    test_video_id, sample_data, data_root,
                )

            with driver.session(database=neo4j_database) as session:
                records = session.execute_read(
                    get_global_sequence_structural_matched_subgraphs, test_video_id
                )

            if records:
                for idx, record in enumerate(records, 1):
                    neo4j_context_text += (
                        f"- Top {idx} Structural Match: Video '{record['video_id']}' "
                        f"(Label: {record['label']})\n"
                        f"  Max nodes matched: {record['max_nodes_matched']}, "
                        f"Total paths: {record['total_matched_sequences']}\n"
                    )
                    for path_idx, det in enumerate(record['match_details'][:2], 1):
                        caps = ' | '.join(det['scene_captions'])

                        # Build concept alignment — dùng label_distribution động
                        concept_flows = []
                        for s_id, c_info in zip(det['sequence_ids'], det['scene_concepts']):
                            if c_info:
                                dist_str = format_concept_distribution(
                                    c_info.get('label_distribution', '')
                                )
                                concept_flows.append(
                                    f"Scene {s_id} -> {c_info['concept_id']} "
                                    f"(Related Videos: {dist_str})"
                                )
                            else:
                                concept_flows.append(f"Scene {s_id} -> No Concept Link")

                        neo4j_context_text += (
                            f"  + Path Sample {path_idx}:\n"
                            f"    Scene Sequence   : {det['sequence_ids']}\n"
                            f"    RST Logic Chain  : [{' -> '.join(det['relation_chain'])}]\n"
                            f"    Semantic Flow    : {caps}\n"
                            f"    Concept Alignment: {' | '.join(concept_flows)}\n"
                        )
                    neo4j_context_text += "\n"
            else:
                neo4j_context_text = "No relevant structural context found in Knowledge Graph.\n"

            # Dọn dẹp đồ thị test ngay sau khi query xong
            with driver.session(database=neo4j_database) as session:
                session.run(
                    "MATCH (v:Video {id: $id}) DETACH DELETE v", id=test_video_id
                )
            print(f"  [CLEANUP] Removed test graph for {test_video_id}.")

            # --------------------------------------------------------------
            # STEP D: Build prompt + đếm token
            # --------------------------------------------------------------
            llm_prompt = build_llm_prompt(
                video_context_text, milvus_context_text, neo4j_context_text
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": llm_prompt},
            ]

            # Đếm token TRƯỚC inference để log và phát hiện prompt quá dài
            n_tokens = count_tokens_gemma(processor, messages)
            print(f"  [TOKEN COUNT] Prompt: {n_tokens:,} tokens (limit: {MAX_CONTEXT_TOKENS:,})")

            # Cắt bớt Neo4j context nếu vượt quá ngưỡng thực tế
            if n_tokens > MAX_CONTEXT_TOKENS:
                print(f"  [WARNING] Prompt exceeds context limit. Truncating Neo4j context.")
                truncated_prompt = build_llm_prompt(
                    video_context_text,
                    milvus_context_text,
                    "Context truncated: prompt exceeded model context window.\n",
                )
                messages[1]["content"] = truncated_prompt
                n_tokens = count_tokens_gemma(processor, messages)
                print(f"  [TOKEN COUNT] After truncation: {n_tokens:,} tokens")

            # --------------------------------------------------------------
            # STEP E: Inference với google/gemma-4-E2B-it
            # --------------------------------------------------------------
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # tắt thinking mode để lấy JSON trực tiếp
            )
            model_inputs = processor(text=text, return_tensors="pt").to(model.device)
            input_len    = model_inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                )

            # Decode chỉ phần output mới (bỏ qua prompt)
            response      = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            parsed_resp   = processor.parse_response(response)
            llm_response  = extract_gemma_response(parsed_resp)

            # --------------------------------------------------------------
            # STEP F: Parse output và lưu kết quả
            # --------------------------------------------------------------
            ground_truth_label = (
                sample_data['y'].item()
                if isinstance(sample_data['y'], torch.Tensor)
                else sample_data['y']
            )
            verdict    = extract_and_parse_json(llm_response)
            pred_label = verdict.get('predicted_label', -1)

            record = {
                "folder_name":    folder,
                "db_test_id":     test_video_id,
                "ground_truth":   ground_truth_label,
                "prediction":     pred_label,
                "token_count":    n_tokens,
                "explanation":    verdict.get('explanation', 'Parsing failed'),
                "raw_llm_output": llm_response,
            }
            evaluation_results.append(record)

            if ground_truth_label == pred_label:
                status = "[MATCHED]"
            elif pred_label == -1:
                status = "[PARSE_ERROR]"
            else:
                status = "[MISMATCHED]"
            print(f"  --> {status} | GT: {ground_truth_label}, Pred: {pred_label}")

            success_count += 1

            # Lưu checkpoint sau mỗi video để có thể resume khi bị gián đoạn
            with open(batch_checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"  [CRITICAL ERROR] {folder}: {e}")
            # Cố gắng dọn dẹp Neo4j dù có lỗi
            try:
                with driver.session(database=neo4j_database) as session:
                    session.run(
                        "MATCH (v:Video {id: $id}) DETACH DELETE v", id=test_video_id
                    )
            except Exception:
                pass
            continue

        finally:
            # Giải phóng bộ nhớ GPU sau mỗi video
            if 'model_inputs' in locals(): del model_inputs
            if 'outputs'      in locals(): del outputs
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.4)

    # ------------------------------------------------------------------
    # Tổng kết pipeline
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"$ PIPELINE COMPLETED. Successfully processed: {success_count} videos.")
    print(f"  Results saved to: {batch_checkpoint_path}")
    print("="*70)

    try:
        driver.close()
        print("[INFO] Neo4j connection closed.")
    except Exception:
        pass


# ==========================================
# 6. ENTRY POINT & ARGUMENT PARSING
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "VideoRAG Binary Inference Pipeline — Milvus + Neo4j + google/gemma-4-E2B-it.\n"
            "Labels: 0 (Low Engagement) and 1 (High Engagement).\n"
            "Milvus and Neo4j credentials are read from environment variables, NOT from args."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Root directory containing per-video folders. "
            "Each folder must have scene_embeddings.pt and segments.json."
        ),
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=True,
        help=(
            "Path to dataset_splits.json. "
            "Must contain a 'test' key mapping to a list of folder names."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=(
            "Path to the output JSON checkpoint file. "
            "Will be created on first run or resumed if it already exists."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "HuggingFace model ID or local path for google/gemma-4-E2B-it. "
            f"Default: {DEFAULT_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--count_tokens_only",
        action="store_true",
        help=(
            "Dry-run mode: measure prompt token length for every video WITHOUT running LLM inference. "
            "Use this BEFORE the first full run to determine the maximum prompt length "
            "in your dataset and decide whether MAX_CONTEXT_TOKENS needs adjustment. "
            "Results are saved to <checkpoint_dir>/token_counts_gemma_binary.json."
        ),
    )

    args = parser.parse_args()
    main(args)