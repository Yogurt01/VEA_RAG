import os
import argparse
import time
import numpy as np
import networkx as nx
import json
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from neo4j import GraphDatabase
from dotenv import load_dotenv
# Sửa nhanh bằng Louvain để Concept có chất lượng cao hơn:
from networkx.community import louvain_communities   # 💡 Cần import này

# Load environment variables from .env file
load_dotenv()

# Retrieve Neo4j credentials
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Validate required environment variables
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
    # query = """
    # CREATE VECTOR INDEX scene_embeddings_idx IF NOT EXISTS
    # FOR (s:Scene) ON (s.embedding)
    # OPTIONS {indexConfig: {
    #   `vector.dimensions`: 2048,
    #   `vector.similarity_function`: 'cosine'
    # }}
    # """
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
    # print("The system is now building the index in the background...")


# def check_vector_index_status(tx):
#     """Check the status of the vector index to ensure it is ready for queries."""
#     query = "SHOW INDEXES YIELD name, type, state, properties WHERE type = 'VECTOR'"
#     result = tx.run(query)
#     records = list(result)

#     if not records:
#         print("WARNING: No Vector Index found. Please ensure index creation was successful.")
#         return

#     for rec in records:
#         print(f"Found Vector Index: '{rec['name']}'")
#         print(f"- State: {rec['state']}")
#         print(f"- Properties: {rec['properties']}")
#         if rec['state'] == 'ONLINE':
#             print("=> Index is ONLINE and ready for queries.")
#         else:
#             print("=> Index is currently building. Queries may be slow or fail until it is ONLINE.")

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


# ==========================================
# 2. GRAPH CONSTRUCTION & ANALYSIS FUNCTIONS
# ==========================================

# def link_similar_scenes_by_label(tx, target_label, top_k, threshold):
#     """
#     Find similar scenes within the same label group using vector search 
#     and create SIMILAR_TO relationships.
#     """
#     cypher_query = """
#     MATCH (src:Scene {video_label: $label})
#     CALL db.index.vector.queryNodes('scene_embeddings_idx', $k, src.embedding)
#     YIELD node AS tgt, score
#     WHERE tgt.video_label = $label
#       AND src.uid <> tgt.uid
#       AND score >= $threshold
#     MERGE (src)-[r:SIMILAR_TO]->(tgt)
#     SET r.cosine_score = score
#     RETURN src.uid AS source, tgt.uid AS target, score
#     """

#     print(f"Calculating and linking similar scenes for Label {target_label} (Top-K: {top_k}, Threshold: {threshold})...")
#     start_time = time.time()

#     result = tx.run(cypher_query, label=target_label, k=top_k, threshold=threshold)
#     records = list(result)

#     end_time = time.time()
#     print(f"Completed in {end_time - start_time:.2f} seconds.")
#     print(f"=> Total new SIMILAR_TO relationships created: {len(records)}")

#     if records:
#         print("\n[LOG] Sample of newly established links:")
#         for rec in records[:3]:
#             print(f"     + {rec['source']} ---> {rec['target']} (Similarity: {rec['score']:.4f})")
#     else:
#         print("[-] No scene pairs exceeded the similarity threshold.")

def link_all_similar_scenes(tx, top_k, threshold):
    """
    PHASE 1 MODIFIED:
    - So sánh tất cả các Scene toàn cục, KHÔNG phân biệt nhãn label 0 hay 1.
    - Loại trừ các cặp Scene thuộc CÙNG MỘT VIDEO (dựa trên mối quan hệ đồ thị với nút gốc :Video).
    """
    cypher_query = """
    MATCH (src:Scene)
    CALL db.index.vector.queryNodes('scene_embeddings_idx', $k, src.embedding)
    YIELD node AS tgt, score
    WHERE src.uid <> tgt.uid
      AND score >= $threshold
      AND NOT (src)<-[:HAS_SCENE]-(:Video)-[:HAS_SCENE]->(tgt)
    MERGE (src)-[r:SIMILAR_TO]->(tgt)
    SET r.cosine_score = score
    RETURN src.uid AS source, tgt.uid AS target, score
    """

    print(f"Calculating and linking similar scenes globally (Top-K: {top_k}, Threshold: {threshold})...")
    start_time = time.time()

    result = tx.run(cypher_query, k=top_k, threshold=threshold)
    records = list(result)

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")
    print(f"=> Total cross-video SIMILAR_TO relationships created: {len(records)}")


# def print_network_summary(tx):
#     """Print a summary of the cross-video similarity network."""
#     query = """
#     MATCH (src:Scene)-[r:SIMILAR_TO]->(tgt:Scene)
#     RETURN src.video_label AS Label, count(r) AS TotalSimilarLinks, avg(r.cosine_score) AS AvgScore
#     ORDER BY Label
#     """
#     result = tx.run(query)
#     print("\n" + "=" * 60)
#     print(" CROSS-VIDEO KNOWLEDGE NETWORK SUMMARY")
#     print("=" * 60)

#     has_data = False
#     for rec in result:
#         has_data = True
#         print(f"Label Group {rec['Label']}:")
#         print(f"- Total SIMILAR_TO links: {rec['TotalSimilarLinks']}")
#         print(f"- Average Cosine Score:   {rec['AvgScore']:.4f}")
#         print("-" * 40)

#     if not has_data:
#         print("No SIMILAR_TO links found. Please verify your threshold and index status.")
#     print("=" * 60 + "\n")

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

# def print_final_database_statistics(tx):
#     """Print detailed statistics of all nodes and relationships in the database."""
#     node_query = """
#     MATCH (n)
#     RETURN labels(n)[0] AS Label, count(n) AS TotalNodes
#     ORDER BY TotalNodes DESC
#     """
#     rel_query = """
#     MATCH ()-[r]->()
#     RETURN type(r) AS Type, count(r) AS TotalRels
#     ORDER BY TotalRels DESC
#     """

#     node_results = list(tx.run(node_query))
#     rel_results = list(tx.run(rel_query))

#     print("\n" + "=" * 60)
#     print(" DETAILED DATABASE STATISTICS")
#     print("=" * 60)

#     print(" NODE COUNTS:")
#     total_nodes = 0
#     for rec in node_results:
#         label_name = rec['Label'] if rec['Label'] else "Unlabeled"
#         print(f"  - [: {label_name}]: {rec['TotalNodes']} nodes")
#         total_nodes += rec['TotalNodes']
#     print(f"  => Total System Nodes: {total_nodes}")

#     print("\n RELATIONSHIP COUNTS:")
#     total_rels = 0
#     for rec in rel_results:
#         print(f"  - [: {rec['Type']}]: {rec['TotalRels']} relationships")
#         total_rels += rec['TotalRels']

#     print("-" * 50)
#     print(f"  => Total System Relationships: {total_rels}")
#     print("=" * 60 + "\n")

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

# def execute_soft_merge_by_label(driver, db_name, target_label):
#     """
#     Fetch scenes of a specific label, cluster them using NetworkX connected components,
#     aggregate their captions and embeddings, and upload as Concept nodes.
#     """
#     scenes_data = {}
#     edges = []

#     print(f"\nInitiating Soft Merge for Label {target_label}...")
    
#     with driver.session(database=db_name) as session:
#         # Fetch node properties (Removed visual_description)
#         nodes_result = session.run(
#             "MATCH (s:Scene {video_label: $label}) "
#             "RETURN s.uid AS uid, s.caption AS caption, s.embedding AS embedding",
#             label=target_label
#         )
#         for rec in nodes_result:
#             scenes_data[rec['uid']] = {
#                 'caption': rec['caption'],
#                 'embedding': rec['embedding']
#             }

#         # Fetch SIMILAR_TO edges
#         edges_result = session.run(
#             "MATCH (src:Scene {video_label: $label})-[r:SIMILAR_TO]->(tgt:Scene {video_label: $label}) "
#             "RETURN src.uid AS source, tgt.uid AS target",
#             label=target_label
#         )
#         for rec in edges_result:
#             edges.append((rec['source'], rec['target']))

#     print(f"-> Loaded {len(scenes_data)} Scene nodes and {len(edges)} similarity edges into memory.")

#     # Build NetworkX graph and find connected components
#     G = nx.Graph()
#     G.add_nodes_from(scenes_data.keys())
#     G.add_edges_from(edges)

#     # clusters = list(nx.connected_components(G))
#     clusters = louvain_communities(G)
    
#     print(f"-> NetworkX algorithm identified {len(clusters)} independent concept clusters.")

#     concepts_to_upload = []

#     for cluster_idx, scene_uids in enumerate(clusters, start=1):
#         uids_list = list(scene_uids)
#         concept_id = f"Concept_L{target_label}_C{cluster_idx}"

#         captions = []
#         embeddings = []

#         for uid in uids_list:
#             node_info = scenes_data[uid]
#             if node_info['caption']: 
#                 captions.append(node_info['caption'])
#             if node_info['embedding']: 
#                 embeddings.append(node_info['embedding'])

#         # Deduplicate text while preserving order
#         merged_caption = " | ".join(list(dict.fromkeys(captions))) if captions else "No caption available"

#         # Calculate mean embedding for the cluster
#         if embeddings:
#             mean_embedding = np.mean(np.array(embeddings), axis=0).tolist()
#         else:
#             mean_embedding = []

#         concepts_to_upload.append({
#             'id': concept_id,
#             'video_label': target_label,
#             'merged_caption': merged_caption,
#             'embedding': mean_embedding,
#             'scene_uids': uids_list
#         })

#     # Upload Concept nodes and BELONGS_TO relationships in a single transaction
#     upload_query = """
#     UNWIND $batch AS c
#     MERGE (con:Concept {id: c.id})
#     SET con.video_label = c.video_label,
#         con.merged_caption = c.merged_caption,
#         con.embedding = c.embedding
#     WITH c, con
#     UNWIND c.scene_uids AS scene_uid
#     MATCH (s:Scene {uid: scene_uid})
#     MERGE (s)-[:BELONGS_TO]->(con)
#     """

#     with driver.session(database=db_name) as session:
#         session.run(upload_query, batch=concepts_to_upload)

#     print(f"[SUCCESS] Created {len(concepts_to_upload)} Concept nodes and established BELONGS_TO relationships.")

def execute_global_soft_merge(driver, db_name):
    """
    PHASE 2 MODIFIED:
    - Gom cụm Louvain toàn cục không chia theo label.
    - Tạo cấu trúc dữ liệu Dictionary chi tiết lưu trong thuộc tính `merged_caption` dưới dạng JSON.
    - Đếm chính xác số lượng video nhãn 0 và nhãn 1 cấu thành nên Concept này.
    """
    scenes_data = {}
    edges = []

    print("\nInitiating Global Soft Merge (Cross-Label Clustering)...")
    
    with driver.session(database=db_name) as session:
        # Lấy thông tin Scene kèm ID của nút Video cha để gom nhóm chính xác
        nodes_result = session.run(
            "MATCH (v:Video)-[:HAS_SCENE]->(s:Scene) "
            "RETURN s.uid AS uid, s.scene_id AS scene_id, s.caption AS caption, "
            "s.embedding AS embedding, s.video_label AS video_label, v.id AS video_id"
        )
        for rec in nodes_result:
            scenes_data[rec['uid']] = {
                'scene_id': rec['scene_id'],
                'caption': rec['caption'],
                'embedding': rec['embedding'],
                'video_label': rec['video_label'],
                'video_id': rec['video_id']
            }

        # Tải toàn bộ các cạnh SIMILAR_TO toàn cục lên RAM
        edges_result = session.run(
            "MATCH (src:Scene)-[r:SIMILAR_TO]->(tgt:Scene) RETURN src.uid AS source, tgt.uid AS target"
        )
        for rec in edges_result:
            edges.append((rec['source'], rec['target']))

    print(f"-> Loaded {len(scenes_data)} Scene nodes and {len(edges)} similarity edges into memory.")

    # Xây dựng đồ thị NetworkX và chạy phân cụm Louvain toàn cục
    G = nx.Graph()
    G.add_nodes_from(scenes_data.keys())
    G.add_edges_from(edges)

    clusters = list(louvain_communities(G))
    print(f"-> Louvain algorithm identified {len(clusters)} independent global concept clusters.")

    concepts_to_upload = []

    for cluster_idx, scene_uids in enumerate(clusters, start=1):
        uids_list = list(scene_uids)
        concept_id = f"Concept_G_C{cluster_idx}" # Định danh dạng Global Cluster

        video_groups = {}
        embeddings = []
        
        # Set dùng để đếm số lượng video duy nhất theo từng label trong cụm này
        vids_label_0 = set()
        vids_label_1 = set()

        for uid in uids_list:
            node_info = scenes_data[uid]
            vid = node_info['video_id']
            v_label = node_info['video_label']
            
            # Phân loại để đếm số lượng video theo nhãn nhị phân
            if v_label == 0:
                vids_label_0.add(vid)
            elif v_label == 1:
                vids_label_1.add(vid)

            # Khởi tạo nhóm nếu video chưa xuất hiện trong cụm này
            if vid not in video_groups:
                video_groups[vid] = {
                    "video_id": vid,
                    "video_label": v_label,
                    "scenes": []
                }
            
            # Thêm thông tin của scene hiện tại vào video tương ứng
            video_groups[vid]["scenes"].append({
                "scene_id": node_info['scene_id'],
                "caption": node_info['caption'] or "No caption available"
            })
            
            if node_info['embedding']:
                embeddings.append(node_info['embedding'])

        # Cấu trúc hóa thông tin chi tiết từng video theo yêu cầu
        structured_videos = []
        for vid, v_info in video_groups.items():
            # Sắp xếp các phân cảnh (scene) tăng dần theo scene_id thực tế
            v_info["scenes"].sort(key=lambda x: x["scene_id"])
            v_info["scene_count_in_this_concept"] = len(v_info["scenes"])
            structured_videos.append(v_info)

        # Đóng gói thành Dictionary lớn tổng thể của Concept
        concept_details = {
            "total_videos_involved": len(video_groups),
            "video_groups": structured_videos
        }
        
        # Chuyển đổi dict thành chuỗi JSON String để lưu trữ an toàn vào thuộc tính Neo4j
        merged_caption_json = json.dumps(concept_details, ensure_ascii=False)

        # Tính toán vector trung bình đại diện cho cụm khái niệm
        if embeddings:
            mean_embedding = np.mean(np.array(embeddings), axis=0).tolist()
        else:
            mean_embedding = []

        concepts_to_upload.append({
            'id': concept_id,
            'merged_caption': merged_caption_json,
            'embedding': mean_embedding,
            'num_videos_label_0': len(vids_label_0),
            'num_videos_label_1': len(vids_label_1),
            'scene_uids': uids_list
        })

    # Đẩy đồng loạt Concept và các liên kết BELONGS_TO lên DB bằng UNWIND
    upload_query = """
    UNWIND $batch AS c
    MERGE (con:Concept {id: c.id})
    SET con.merged_caption = c.merged_caption,
        con.embedding = c.embedding,
        con.num_videos_label_0 = c.num_videos_label_0,
        con.num_videos_label_1 = c.num_videos_label_1
    WITH c, con
    UNWIND c.scene_uids AS scene_uid
    MATCH (s:Scene {uid: scene_uid})
    MERGE (s)-[:BELONGS_TO]->(con)
    """

    with driver.session(database=db_name) as session:
        session.run(upload_query, batch=concepts_to_upload)

    print(f"[SUCCESS] Created {len(concepts_to_upload)} global Concept nodes with structural dictionary profiles.")

# ==========================================
# 4. PHASE 3: METRIC ANALYSIS REPORT (UPDATED)
# ==========================================

# def print_soft_merge_summary(tx):
#     """Print a summary of the generated Concept network."""
#     # query = """
#     # MATCH (c:Concept)
#     # RETURN c.video_label AS Label,
#     #        count(c) AS TotalConcepts,
#     #        sum(COUNT { (c)<-[:BELONGS_TO]-(:Scene) }) AS TotalLinkedScenes,
#     #        avg(COUNT { (c)<-[:BELONGS_TO]-(:Scene) }) AS AvgScenesPerConcept
#     # ORDER BY Label
#     # """
#     query = """
#     MATCH (c:Concept)
#     OPTIONAL MATCH (c)<-[:BELONGS_TO]-(s:Scene)
#     WITH c, count(s) AS num_scenes
#     RETURN c.video_label AS Label,
#            count(DISTINCT c) AS TotalConcepts,
#            sum(num_scenes) AS TotalLinkedScenes,
#            avg(num_scenes) AS AvgScenesPerConcept
#     ORDER BY Label
#     """
#     result = tx.run(query)
#     print("\n" + "=" * 60)
#     print(" CONCEPT NETWORK SUMMARY (SOFT MERGE RESULTS)")
#     print("=" * 60)

#     has_data = False
#     for rec in result:
#         has_data = True
#         print(f"Label Group {rec['Label']}:")
#         print(f"  - Total Concept nodes created:        {rec['TotalConcepts']}")
#         print(f"  - Total Scene nodes successfully clustered: {rec['TotalLinkedScenes']}")
#         print(f"  - Average scenes per concept:         {rec['AvgScenesPerConcept']:.2f}")
#         print("-" * 40)

#     if not has_data:
#         print("No Concept data found. Please verify the Soft Merge execution.")
#     print("=" * 60 + "\n")

def print_soft_merge_summary(tx):
    """
    PHASE 3 MODIFIED:
    - Báo cáo tổng số cụm khái niệm được sinh ra toàn hệ thống.
    - Thống kê phân phối kích cỡ: số node trung bình, lớn nhất, nhỏ nhất trong 1 cụm.
    """
    query = """
    MATCH (c:Concept)
    OPTIONAL MATCH (c)<-[:BELONGS_TO]-(s:Scene)
    WITH c, count(s) AS num_scenes
    RETURN count(DISTINCT c) AS TotalConcepts,
           sum(num_scenes) AS TotalLinkedScenes,
           avg(num_scenes) AS AvgScenesPerConcept,
           max(num_scenes) AS MaxScenesPerConcept,
           min(num_scenes) AS MinScenesPerConcept
    """
    result = tx.run(query).single()
    print("\n" + "=" * 60)
    print(" GLOBAL CONCEPT NETWORK SUMMARY (PHASE 3 REPORT)")
    print("=" * 60)

    if result and result['TotalConcepts'] > 0:
        print(f" Tổng số cụm khái niệm (Concept) tạo thành: {result['TotalConcepts']}")
        print(f" Tổng số phân cảnh (Scene) được gom cụm   : {result['TotalLinkedScenes']}")
        print(f" Số lượng Node (Scene) trung bình / cụm   : {result['AvgScenesPerConcept']:.2f} nodes")
        print(f" Kích thước cụm LỚN NHẤT (Max Nodes)       : {result['MaxScenesPerConcept']} nodes")
        print(f" Kích thước cụm NHỎ NHẤT (Min Nodes)       : {result['MinScenesPerConcept']} nodes")
    else:
        print("Không tìm thấy dữ liệu Concept. Hãy kiểm tra lại hàm Soft Merge.")
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

            # Phase 1: Indexing and Similarity Linking
            print("--- PHASE 1: Vector Indexing and Similarity Linking ---")
            with driver.session(database=DATABASE) as session:
                session.execute_write(create_scene_vector_index)
                session.execute_read(check_vector_index_status)
                
                # for label in args.labels:
                #     session.execute_write(
                #         # link_similar_scenes_by_label, 
                #         link_all_similar_scenes,
                #         target_label=label, 
                #         top_k=args.top_k, 
                #         threshold=args.threshold
                #     )
                
                # Sửa đổi: Chạy hàm liên kết toàn cục, không tách label loop nữa
                session.execute_write(
                    link_all_similar_scenes, 
                    top_k=args.top_k, 
                    threshold=args.threshold
                )

                session.execute_read(print_network_summary)
                session.execute_read(print_final_database_statistics)

            # Phase 2: Soft Merge (Concept Clustering)
            print("\n--- PHASE 2: Soft Merge (Concept Clustering) ---")
            # for label in args.labels:
            #     execute_soft_merge_by_label(driver, DATABASE, target_label=label)
            execute_global_soft_merge(driver, DATABASE)

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
        "--labels", 
        type=int, 
        nargs="+", 
        default=[0, 1], 
        help="List of target labels to process (e.g., --labels 0 1). Default: 0 1"
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
    
    args = parser.parse_args()
    main(args)
