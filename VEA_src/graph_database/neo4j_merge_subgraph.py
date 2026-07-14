import os
import argparse
import time
import json
import numpy as np
import networkx as nx
from neo4j import GraphDatabase
from dotenv import load_dotenv
from networkx.algorithms.community import louvain_communities
from collections import defaultdict
from llm_utils import QwenGenerator, generate_concept_description

# Load environment variables from .env file
load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not all([URI, USERNAME, PASSWORD]):
    raise ValueError(
        "Missing required Neo4j environment variables. "
        "Please ensure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are set in your .env file."
    )


# ==========================================
# 1. VECTOR INDEXING FUNCTIONS
# ==========================================

def create_scene_vector_index(tx):
    """Create a vector index for Scene embeddings if it does not already exist."""
    query = """
    CREATE VECTOR INDEX scene_embeddings_idx IF NOT EXISTS
    FOR (s:Scene) ON (s.embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 2048,
        `vector.similarity_function`: "cosine"
      }
    }
    """
    tx.run(query)
    print("Vector index creation command sent successfully to Neo4j.")


def check_vector_index_status(tx):
    """Check the status of the vector index."""
    query = "SHOW INDEXES YIELD name, type, state, properties WHERE type = 'VECTOR'"
    result = tx.run(query)
    for rec in result:
        print(f"Found Vector Index: '{rec['name']}' | State: {rec['state']}")


def create_concept_vector_index(tx):
    """Create a vector index for Concept embeddings."""
    query = """
    CREATE VECTOR INDEX concept_embeddings_idx IF NOT EXISTS
    FOR (c:Concept) ON (c.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 2048,
      `vector.similarity_function`: 'cosine'
    }}
    """
    tx.run(query)
    print("Successfully initialized Vector Index 'concept_embeddings_idx'.")

def clear_similarity_relationships(tx):
    tx.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")
    print("Cleared all SIMILAR_TO relationships.")

def clear_concepts(tx):
    tx.run("MATCH (c:Concept) DETACH DELETE c")
    print("Cleared all Concept nodes and relationships.")

# ==========================================
# 2. GRAPH CONSTRUCTION & ANALYSIS FUNCTIONS
# ==========================================

def link_all_similar_scenes(tx, top_k: int, threshold: float):
    """
    Retrieve Top-(top_k * multiplier) nearest neighbors for every Scene.
    Do NOT create relationships here.

    The returned candidates will later be filtered in Python to:
        - remove same-video scenes
        - keep only Top-K cross-video neighbors
    """

    internal_k = top_k * 7

    cypher_query = """
    MATCH (src:Scene)<-[:HAS_SCENE]-(srcVideo:Video)

    CALL db.index.vector.queryNodes(
        'scene_embeddings_idx',
        $internal_k,
        src.embedding
    )

    YIELD node AS tgt, score

    MATCH (tgt)<-[:HAS_SCENE]-(tgtVideo:Video)

    WHERE
        src.uid <> tgt.uid
        AND score >= $threshold

    RETURN
        src.uid      AS source_uid,
        srcVideo.id  AS source_video,
        tgt.uid      AS target_uid,
        tgtVideo.id  AS target_video,
        score

    ORDER BY source_uid, score DESC
    """

    print(
        f"Searching Top-{internal_k} candidates for each Scene "
        f"(Final Top-K = {top_k}, Threshold = {threshold})..."
    )

    start = time.time()

    result = tx.run(
        cypher_query,
        internal_k=internal_k,
        threshold=threshold
    )

    records = [dict(r) for r in result]

    elapsed = time.time() - start

    print(
        f"Retrieved {len(records)} candidate neighbors "
        f"in {elapsed:.2f} seconds."
    )

    return records

def create_similarity_relationships(driver,
                                    db_name,
                                    candidate_records,
                                    top_k):
    """
    Filter candidate neighbors in Python and batch-create SIMILAR_TO
    relationships.

    Steps
    -----
    1. Remove same-video neighbors.
    2. Keep Top-K remaining neighbors.
    3. Batch MERGE relationships.
    """

    grouped = defaultdict(list)

    for rec in candidate_records:

        if rec["source_video"] == rec["target_video"]:
            continue

        grouped[rec["source_uid"]].append(rec)

    relationships = []

    for src_uid, neighbors in grouped.items():
        neighbors.sort(
            key=lambda x: x["score"],
            reverse=True
        )
        seen_targets = set()
        for n in neighbors[:top_k]:
            if n["target_uid"] in seen_targets:
                continue
            seen_targets.add(n["target_uid"])
            relationships.append({
                "src_uid": src_uid,
                "tgt_uid": n["target_uid"],
                "score": float(n["score"])
            })

    print(
        f"Creating {len(relationships)} cross-video "
        f"SIMILAR_TO relationships..."
    )

    query = """
    UNWIND $batch AS rel

    MATCH (src:Scene {uid: rel.src_uid})
    MATCH (tgt:Scene {uid: rel.tgt_uid})

    MERGE (src)-[r:SIMILAR_TO]->(tgt)

    SET r.cosine_score = rel.score
    """

    with driver.session(database=db_name) as session:

        session.run(
            query,
            batch=relationships
        )

    print("Finished creating SIMILAR_TO relationships.")


def print_network_summary(tx):
    """Print a summary of the global similarity network."""
    query = """
    MATCH (src:Scene)-[r:SIMILAR_TO]->(tgt:Scene)
    RETURN count(r) AS TotalSimilarLinks, avg(r.cosine_score) AS AvgScore
    """
    result = tx.run(query).single()
    print("\n" + "=" * 60)
    print(" CROSS-VIDEO KNOWLEDGE NETWORK SUMMARY")
    print("=" * 60)
    if result:
        print(f"- Total Global SIMILAR_TO links: {result['TotalSimilarLinks']}")
        print(f"- Average Global Cosine Score:   {result['AvgScore']:.4f}")
    print("=" * 60 + "\n")


def print_final_database_statistics(tx):
    """Print statistics of all nodes and relationships."""
    node_query = "MATCH (n) RETURN labels(n)[0] AS Label, count(n) AS TotalNodes ORDER BY TotalNodes DESC"
    rel_query = "MATCH ()-[r]->() RETURN type(r) AS Type, count(r) AS TotalRels ORDER BY TotalRels DESC"

    print(" DETAILED DATABASE STATISTICS:")
    for rec in tx.run(node_query):
        print(f"  - [: {rec['Label']}]: {rec['TotalNodes']} nodes")
    for rec in tx.run(rel_query):
        print(f"  - [: {rec['Type']}]: {rec['TotalRels']} relationships")
    print("-" * 40)


# ==========================================
# 3. SOFT MERGE (CONCEPT CLUSTERING) FUNCTIONS
# ==========================================

def build_concept_prompt(video_groups, max_scenes_per_video=3, max_videos=5):
    """
    Build a textual prompt for one Concept cluster.

    This prompt will later be sent to an LLM
    to summarize the semantic concept.
    """
    # Sắp xếp video theo số lượng scene giảm dần
    sorted_videos = sorted(video_groups, key=lambda x: len(x["scenes"]), reverse=True)
    # Chỉ lấy tối đa max_videos video
    selected_videos = sorted_videos[:max_videos]

    lines = []

    lines.append(
        "You are given a cluster of semantically similar scenes "
        "coming from multiple videos."
    )

    lines.append(
        "Analyze the common concept shared by these scenes."
    )
    lines.append("Some scenes may come from different videos but describe the same underlying concept. Focus on the shared concept instead of individual events.")
    lines.append("")

    # for video in sorted(
    #     video_groups,
    #     key=lambda x: x["video_id"]
    # ):
    for video in selected_videos:
        scenes = video["scenes"][:max_scenes_per_video]
        lines.append("=" * 70)
        lines.append(f"VIDEO: {video['video_id']} (Label: {video['video_label']})")
        lines.append("Discourse Summary:")
        lines.append(video["rst_summary"])
        lines.append("")
        # for scene in video["scenes"]:
        for scene in scenes:
            lines.append(
                f"### Scene {scene['scene_id']}"
            )
            lines.append(
                f"Caption: {scene['caption']}"
            )
            lines.append(
                f"Visual Elements: {scene['visual_elements']}"
            )
            lines.append(
                f"Audio Tags: {scene['audio_tags']}"
            )
            lines.append(
                f"Audio Vibes: {scene['audio_vibes']}"
            )
            lines.append("")
    lines.append("=" * 70)
    lines.append("Please summarize this concept and return your answer in **JSON format** with the following keys:")
    lines.append("{")
    lines.append('  "summary": "overall semantic concept",')
    lines.append('  "visual_style": "visual characteristics",')
    lines.append('  "audio_style": "audio characteristics",')
    lines.append('  "storyline": "narrative flow",')
    lines.append('  "keywords": ["keyword1", "keyword2", ...]')
    lines.append("}")
    lines.append("Do not include any extra text outside the JSON.")
    return "\n".join(lines)

def execute_global_soft_merge(driver, db_name, skip_llm=False, generator=None, resolution=1.0, min_scenes=2):
    """
    Perform global Louvain clustering across all labels.
    - Builds a structured JSON dictionary for cluster_metadata.
    - Dynamically counts contributing videos per label (supports any number of labels).
    - Stores label distribution as a JSON string property on each Concept node.
    """
    scenes_data = {}
    edges = []

    print("\nInitiating Global Soft Merge (Cross-Label Clustering)...")

    with driver.session(database=db_name) as session:
        nodes_result = session.run(
            """
            MATCH (v:Video)-[:HAS_SCENE]->(s:Scene)

            RETURN

                s.uid               AS uid,

                s.scene_id          AS scene_id,

                s.caption           AS caption,

                s.visual_elements   AS visual_elements,

                s.audio_tags        AS audio_tags,

                s.audio_vibes       AS audio_vibes,

                s.embedding         AS embedding,

                s.video_label       AS video_label,

                v.id                AS video_id,

                v.rst_summary       AS rst_summary
            """
        )
        for rec in nodes_result:
            scenes_data[rec['uid']] = {
                'scene_id': rec['scene_id'],
                'caption': rec['caption'],
                'visual_elements': rec['visual_elements'],
                'audio_tags': rec['audio_tags'],
                'audio_vibes': rec['audio_vibes'],
                'embedding': rec['embedding'],
                'video_label': rec['video_label'],
                'video_id': rec['video_id'],
                'rst_summary': rec['rst_summary']
            }

        # edges_result = session.run(
        #     "MATCH (src:Scene)-[r:SIMILAR_TO]->(tgt:Scene) RETURN src.uid AS source, tgt.uid AS target"
        # )
        # Sửa thành:
        edges_result = session.run("""
            MATCH (src:Scene)-[r:SIMILAR_TO]->(tgt:Scene)
            RETURN src.uid AS source, tgt.uid AS target, r.cosine_score AS score
        """)
        for rec in edges_result:
            edges.append((rec['source'], rec['target'], rec['score']))

    print(f"-> Loaded {len(scenes_data)} Scene nodes and {len(edges)} similarity edges into memory.")

    G = nx.Graph()
    G.add_nodes_from(scenes_data.keys())
    # G.add_edges_from(edges)
    for src, tgt, score in edges:
        G.add_edge(src, tgt, weight=score)

    # clusters = list(louvain_communities(G))
    clusters = list(louvain_communities(G, weight='weight', resolution=resolution, seed=42))
    clusters.sort(key=len, reverse=True)
    print(f"-> Louvain algorithm identified {len(clusters)} independent global concept clusters.")

    concepts_to_upload = []

    for cluster_idx, scene_uids in enumerate(clusters, start=1):
        uids_list = list(scene_uids)
        if len(uids_list) < min_scenes:
            # print(f"   Skipping cluster {cluster_idx} with only {len(uids_list)} scenes.")
            continue  # bỏ qua
        concept_id = f"Concept_G_C{cluster_idx}"

        video_groups = {}
        embeddings = []

        # Dynamic label counter: { label_value: set of video_ids }
        vids_by_label = {}

        for uid in uids_list:
            node_info = scenes_data[uid]
            vid = node_info['video_id']
            v_label = node_info['video_label']

            # Dynamically track unique videos per label
            if v_label not in vids_by_label:
                vids_by_label[v_label] = set()
            vids_by_label[v_label].add(vid)

            if vid not in video_groups:
                video_groups[vid] = {
                    "video_id": vid,
                    "video_label": v_label,
                    "rst_summary": node_info['rst_summary'],
                    "scenes": []
                }

            video_groups[vid]["scenes"].append({
                "scene_id": node_info['scene_id'],
                "caption": node_info['caption'] or "No caption available",
                "visual_elements": node_info['visual_elements'],
                "audio_tags": node_info['audio_tags'],
                "audio_vibes": node_info['audio_vibes']
            })

            if node_info['embedding']:
                embeddings.append(node_info['embedding'])

        structured_videos = []
        for vid, v_info in video_groups.items():
            v_info["scenes"].sort(key=lambda x: x["scene_id"])
            v_info["scene_count_in_this_concept"] = len(v_info["scenes"])
            structured_videos.append(v_info)

        concept_prompt = build_concept_prompt(structured_videos, max_videos=5, max_scenes_per_video=3)

        if cluster_idx <= 3:
            print()
            print("=" * 100)
            print(f"Prompt Preview ({concept_id})")
            print("=" * 100)
            print(concept_prompt)
            print("=" * 100)

        if skip_llm or generator is None:
            llm_output = {"summary": "", "visual_style": "", "audio_style": "", "storyline": "", "keywords": []}
        else:
            llm_output = generate_concept_description(concept_prompt, generator)

        concept_details = {
            "total_videos_involved": len(video_groups),
            "video_groups": structured_videos
        }
        cluster_metadata_json = json.dumps(concept_details, ensure_ascii=False)

        if embeddings:
            mean_embedding = np.mean(np.array(embeddings), axis=0)
            norm = np.linalg.norm(mean_embedding) + 1e-12
            mean_embedding = (mean_embedding / norm).tolist()
        else:
            mean_embedding = []

        # Build dynamic label distribution: { "num_videos_label_0": N, "num_videos_label_1": M, ... }
        # label_distribution = {
        #     f"num_videos_label_{label}": len(vid_set)
        #     for label, vid_set in vids_by_label.items()
        # }
        # label_distribution_json = json.dumps(
        #     {str(k): v for k, v in sorted(vids_by_label.items(), key=lambda x: x[0])
        #      if True  # just converting set sizes
        #      },
        #     default=lambda s: len(s) if isinstance(s, set) else s
        # )
        # Simpler and cleaner version
        label_distribution_json = json.dumps(
            {str(label): len(vid_set) for label, vid_set in sorted(vids_by_label.items())},
            ensure_ascii=False
        )

        max_scenes = max([len(v['scenes']) for v in structured_videos]) if structured_videos else 0
        cluster_statistics = {
            "num_scenes": len(uids_list),
            "num_videos": len(video_groups),
            "num_labels": len(vids_by_label),
            "largest_video_contribution": max_scenes
        }

        concepts_to_upload.append({
            'id': concept_id,
            'cluster_metadata': cluster_metadata_json,
            'embedding': mean_embedding,
            'label_distribution': label_distribution_json,
            'concept_prompt': concept_prompt,
            'summary': llm_output.get('summary', ''),
            'visual_style': llm_output.get('visual_style', ''),
            'audio_style': llm_output.get('audio_style', ''),
            'storyline': llm_output.get('storyline', ''),
            'keywords': llm_output.get('keywords', []),
            'scene_uids': uids_list,
            'cluster_statistics': json.dumps(cluster_statistics, ensure_ascii=False),
        })

    upload_query = """
    UNWIND $batch AS c
    MERGE (con:Concept {id: c.id})
    SET con.cluster_metadata    = c.cluster_metadata,
        con.embedding           = c.embedding,
        con.label_distribution  = c.label_distribution,
        con.concept_prompt      = c.concept_prompt,
        con.summary             = c.summary,
        con.visual_style        = c.visual_style,
        con.audio_style         = c.audio_style,
        con.storyline           = c.storyline,
        con.keywords            = c.keywords,
        con.cluster_statistics  = c.cluster_statistics
    WITH c, con
    UNWIND c.scene_uids AS scene_uid
    MATCH (s:Scene {uid: scene_uid})
    MERGE (s)-[:BELONGS_TO]->(con)
    """

    with driver.session(database=db_name) as session:
        session.run(upload_query, batch=concepts_to_upload)

    print(f"[SUCCESS] Created {len(concepts_to_upload)} global Concept nodes with structural dictionary profiles.")


# ==========================================
# 4. PHASE 3: METRIC ANALYSIS REPORT
# ==========================================

def print_soft_merge_summary(tx):
    """
    Report on total concept clusters generated.
    Includes cluster size distribution: avg, max, min scenes per cluster.
    """
    query = """
    MATCH (c:Concept)
    OPTIONAL MATCH (c)<-[:BELONGS_TO]-(s:Scene)
    WITH c, count(s) AS num_scenes
    RETURN count(DISTINCT c) AS TotalConcepts,
           sum(num_scenes)   AS TotalLinkedScenes,
           avg(num_scenes)   AS AvgScenesPerConcept,
           max(num_scenes)   AS MaxScenesPerConcept,
           min(num_scenes)   AS MinScenesPerConcept
    """
    result = tx.run(query).single()
    print("\n" + "=" * 60)
    print(" GLOBAL CONCEPT NETWORK SUMMARY (PHASE 3 REPORT)")
    print("=" * 60)

    if result and result['TotalConcepts'] > 0:
        print(f" Total Concept clusters created     : {result['TotalConcepts']}")
        print(f" Total Scene nodes clustered        : {result['TotalLinkedScenes']}")
        print(f" Average Scenes per Concept         : {result['AvgScenesPerConcept']:.2f}")
        print(f" Largest cluster  (Max Scenes)      : {result['MaxScenesPerConcept']}")
        print(f" Smallest cluster (Min Scenes)      : {result['MinScenesPerConcept']}")
    else:
        print("No Concept data found. Please verify the Soft Merge execution.")
    print("=" * 60 + "\n")


# ==========================================
# 5. MAIN EXECUTION LOGIC
# ==========================================

def main(args):
    print("Initializing Neo4j Graph Construction Pipeline...")

    try:
        with GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD)) as driver:
            driver.verify_connectivity()
            print("Successfully connected to Neo4j.\n")

            if args.clean:
                with driver.session(database=DATABASE) as session:
                    session.execute_write(clear_similarity_relationships)
                    session.execute_write(clear_concepts)

            # Phase 1: Indexing and Similarity Linking
            print("--- PHASE 1: Vector Indexing and Similarity Linking ---")
            with driver.session(database=DATABASE) as session:
                session.execute_write(create_scene_vector_index)
                session.execute_read(check_vector_index_status)
                candidate_records = session.execute_write(
                    link_all_similar_scenes,
                    top_k=args.top_k,
                    threshold=args.threshold
                )

            create_similarity_relationships(
                driver=driver,
                db_name=DATABASE,
                candidate_records=candidate_records,
                top_k=args.top_k
            )

            with driver.session(database=DATABASE) as session:
                session.execute_read(print_network_summary)
                session.execute_read(print_final_database_statistics)

            # Phase 2: Soft Merge (Concept Clustering)
            print("\n--- PHASE 2: Soft Merge (Global Concept Clustering) ---")
            # execute_global_soft_merge(driver, DATABASE)
            generator = None
            if not args.skip_llm:
                print("Loading Qwen model from", args.model_path)
                try:
                    generator = QwenGenerator(args.model_path)
                    print("Model loaded successfully.")
                except Exception as e:
                    print(f"Failed to load model: {e}")
                    print("Continuing without LLM generation.")
                    args.skip_llm = True
            execute_global_soft_merge(driver, DATABASE, args.skip_llm, generator, args.resolution, args.min_scenes)

            # Phase 3: Concept Indexing and Final Summary
            print("\n--- PHASE 3: Concept Indexing and Final Summary ---")
            with driver.session(database=DATABASE) as session:
                session.execute_write(create_concept_vector_index)
                session.execute_read(print_soft_merge_summary)

    except Exception as e:
        print(f"Critical Error: Failed to connect or process data in Neo4j: {e}")
        return

    print("\nPipeline execution completed successfully.")


# ==========================================
# 6. ENTRY POINT & ARGUMENT PARSING
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge subgraphs and build a concept network in a Neo4j Database."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top similar nodes to retrieve per vector search. Default: 5"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for creating SIMILAR_TO relationships. Default: 0.85"
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Skip LLM concept generation."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/models/Qwen3-4B-Instruct-2507",
        help="Path to the local Qwen model directory."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing SIMILAR_TO relationships and Concept nodes before running."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for Louvain clustering. Higher = more clusters, lower = fewer clusters. Default: 1.0"
    )
    parser.add_argument(
        "--min_scenes",
        type=int,
        default=2,
        help="Minimum number of scenes required to form a Concept cluster. Clusters smaller than this are ignored."
    )

    args = parser.parse_args()
    main(args)
