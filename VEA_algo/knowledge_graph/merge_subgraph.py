import os
import argparse
import time
import numpy as np
import networkx as nx
import torch
from neo4j import GraphDatabase
from dotenv import load_dotenv

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
    query = """
    CREATE VECTOR INDEX scene_embeddings_idx IF NOT EXISTS
    FOR (s:Scene) ON (s.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 2048,
      `vector.similarity_function`: 'cosine'
    }}
    """
    tx.run(query)
    print("Vector index creation command sent successfully to Neo4j.")
    print("The system is now building the index in the background...")


def check_vector_index_status(tx):
    """Check the status of the vector index to ensure it is ready for queries."""
    query = "SHOW INDEXES YIELD name, type, state, properties WHERE type = 'VECTOR'"
    result = tx.run(query)
    records = list(result)

    if not records:
        print("WARNING: No Vector Index found. Please ensure index creation was successful.")
        return

    for rec in records:
        print(f"Found Vector Index: '{rec['name']}'")
        print(f"- State: {rec['state']}")
        print(f"- Properties: {rec['properties']}")
        if rec['state'] == 'ONLINE':
            print("=> Index is ONLINE and ready for queries.")
        else:
            print("=> Index is currently building. Queries may be slow or fail until it is ONLINE.")


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
    print("Successfully initialized Vector Index 'concept_embeddings_idx' for Concept nodes.")


# ==========================================
# 2. GRAPH CONSTRUCTION & ANALYSIS FUNCTIONS
# ==========================================

def link_similar_scenes_by_label(tx, target_label, top_k, threshold):
    """
    Find similar scenes within the same label group using vector search 
    and create SIMILAR_TO relationships.
    """
    cypher_query = """
    MATCH (src:Scene {video_label: $label})
    CALL db.index.vector.queryNodes('scene_embeddings_idx', $k, src.embedding)
    YIELD node AS tgt, score
    WHERE tgt.video_label = $label
      AND src.uid <> tgt.uid
      AND score >= $threshold
    MERGE (src)-[r:SIMILAR_TO]->(tgt)
    SET r.cosine_score = score
    RETURN src.uid AS source, tgt.uid AS target, score
    """

    print(f"Calculating and linking similar scenes for Label {target_label} (Top-K: {top_k}, Threshold: {threshold})...")
    start_time = time.time()

    result = tx.run(cypher_query, label=target_label, k=top_k, threshold=threshold)
    records = list(result)

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")
    print(f"=> Total new SIMILAR_TO relationships created: {len(records)}")

    if records:
        print("\n[LOG] Sample of newly established links:")
        for rec in records[:3]:
            print(f"     + {rec['source']} ---> {rec['target']} (Similarity: {rec['score']:.4f})")
    else:
        print("[-] No scene pairs exceeded the similarity threshold.")


def print_network_summary(tx):
    """Print a summary of the cross-video similarity network."""
    query = """
    MATCH (src:Scene)-[r:SIMILAR_TO]->(tgt:Scene)
    RETURN src.video_label AS Label, count(r) AS TotalSimilarLinks, avg(r.cosine_score) AS AvgScore
    ORDER BY Label
    """
    result = tx.run(query)
    print("\n" + "=" * 60)
    print(" CROSS-VIDEO KNOWLEDGE NETWORK SUMMARY")
    print("=" * 60)

    has_data = False
    for rec in result:
        has_data = True
        print(f"Label Group {rec['Label']}:")
        print(f"- Total SIMILAR_TO links: {rec['TotalSimilarLinks']}")
        print(f"- Average Cosine Score:   {rec['AvgScore']:.4f}")
        print("-" * 40)

    if not has_data:
        print("No SIMILAR_TO links found. Please verify your threshold and index status.")
    print("=" * 60 + "\n")


def print_final_database_statistics(tx):
    """Print detailed statistics of all nodes and relationships in the database."""
    node_query = """
    MATCH (n)
    RETURN labels(n)[0] AS Label, count(n) AS TotalNodes
    ORDER BY TotalNodes DESC
    """
    rel_query = """
    MATCH ()-[r]->()
    RETURN type(r) AS Type, count(r) AS TotalRels
    ORDER BY TotalRels DESC
    """

    node_results = list(tx.run(node_query))
    rel_results = list(tx.run(rel_query))

    print("\n" + "=" * 60)
    print(" DETAILED DATABASE STATISTICS")
    print("=" * 60)

    print(" NODE COUNTS:")
    total_nodes = 0
    for rec in node_results:
        label_name = rec['Label'] if rec['Label'] else "Unlabeled"
        print(f"  - [: {label_name}]: {rec['TotalNodes']} nodes")
        total_nodes += rec['TotalNodes']
    print(f"  => Total System Nodes: {total_nodes}")

    print("\n RELATIONSHIP COUNTS:")
    total_rels = 0
    for rec in rel_results:
        print(f"  - [: {rec['Type']}]: {rec['TotalRels']} relationships")
        total_rels += rec['TotalRels']

    print("-" * 50)
    print(f"  => Total System Relationships: {total_rels}")
    print("=" * 60 + "\n")


# ==========================================
# 3. SOFT MERGE (CONCEPT CLUSTERING) FUNCTIONS
# ==========================================

def execute_soft_merge_by_label(driver, db_name, target_label):
    """
    Fetch scenes of a specific label, cluster them using NetworkX connected components,
    aggregate their captions and embeddings, and upload as Concept nodes.
    """
    scenes_data = {}
    edges = []

    print(f"\nInitiating Soft Merge for Label {target_label}...")
    
    with driver.session(database=db_name) as session:
        # Fetch node properties (Removed visual_description)
        nodes_result = session.run(
            "MATCH (s:Scene {video_label: $label}) "
            "RETURN s.uid AS uid, s.caption AS caption, s.embedding AS embedding",
            label=target_label
        )
        for rec in nodes_result:
            scenes_data[rec['uid']] = {
                'caption': rec['caption'],
                'embedding': rec['embedding']
            }

        # Fetch SIMILAR_TO edges
        edges_result = session.run(
            "MATCH (src:Scene {video_label: $label})-[r:SIMILAR_TO]->(tgt:Scene {video_label: $label}) "
            "RETURN src.uid AS source, tgt.uid AS target",
            label=target_label
        )
        for rec in edges_result:
            edges.append((rec['source'], rec['target']))

    print(f"-> Loaded {len(scenes_data)} Scene nodes and {len(edges)} similarity edges into memory.")

    # Build NetworkX graph and find connected components
    G = nx.Graph()
    G.add_nodes_from(scenes_data.keys())
    G.add_edges_from(edges)

    clusters = list(nx.connected_components(G))
    print(f"-> NetworkX algorithm identified {len(clusters)} independent concept clusters.")

    concepts_to_upload = []

    for cluster_idx, scene_uids in enumerate(clusters, start=1):
        uids_list = list(scene_uids)
        concept_id = f"Concept_L{target_label}_C{cluster_idx}"

        captions = []
        embeddings = []

        for uid in uids_list:
            node_info = scenes_data[uid]
            if node_info['caption']: 
                captions.append(node_info['caption'])
            if node_info['embedding']: 
                embeddings.append(node_info['embedding'])

        # Deduplicate text while preserving order
        merged_caption = " | ".join(list(dict.fromkeys(captions))) if captions else "No caption available"

        # Calculate mean embedding for the cluster
        if embeddings:
            mean_embedding = np.mean(np.array(embeddings), axis=0).tolist()
        else:
            mean_embedding = []

        concepts_to_upload.append({
            'id': concept_id,
            'video_label': target_label,
            'merged_caption': merged_caption,
            'embedding': mean_embedding,
            'scene_uids': uids_list
        })

    # Upload Concept nodes and BELONGS_TO relationships in a single transaction
    upload_query = """
    UNWIND $batch AS c
    MERGE (con:Concept {id: c.id})
    SET con.video_label = c.video_label,
        con.merged_caption = c.merged_caption,
        con.embedding = c.embedding
    WITH c, con
    UNWIND c.scene_uids AS scene_uid
    MATCH (s:Scene {uid: scene_uid})
    MERGE (s)-[:BELONGS_TO]->(con)
    """

    with driver.session(database=db_name) as session:
        session.run(upload_query, batch=concepts_to_upload)

    print(f"[SUCCESS] Created {len(concepts_to_upload)} Concept nodes and established BELONGS_TO relationships.")


def print_soft_merge_summary(tx):
    """Print a summary of the generated Concept network."""
    query = """
    MATCH (c:Concept)
    RETURN c.video_label AS Label,
           count(c) AS TotalConcepts,
           sum(COUNT { (c)<-[:BELONGS_TO]-(:Scene) }) AS TotalLinkedScenes,
           avg(COUNT { (c)<-[:BELONGS_TO]-(:Scene) }) AS AvgScenesPerConcept
    ORDER BY Label
    """
    result = tx.run(query)
    print("\n" + "=" * 60)
    print(" CONCEPT NETWORK SUMMARY (SOFT MERGE RESULTS)")
    print("=" * 60)

    has_data = False
    for rec in result:
        has_data = True
        print(f"Label Group {rec['Label']}:")
        print(f"  - Total Concept nodes created:        {rec['TotalConcepts']}")
        print(f"  - Total Scene nodes successfully clustered: {rec['TotalLinkedScenes']}")
        print(f"  - Average scenes per concept:         {rec['AvgScenesPerConcept']:.2f}")
        print("-" * 40)

    if not has_data:
        print("No Concept data found. Please verify the Soft Merge execution.")
    print("=" * 60 + "\n")


# ==========================================
# 4. MAIN EXECUTION LOGIC
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
                
                for label in args.labels:
                    session.execute_write(
                        link_similar_scenes_by_label, 
                        target_label=label, 
                        top_k=args.top_k, 
                        threshold=args.threshold
                    )
                
                session.execute_read(print_network_summary)
                session.execute_read(print_final_database_statistics)

            # Phase 2: Soft Merge (Concept Clustering)
            print("\n--- PHASE 2: Soft Merge (Concept Clustering) ---")
            for label in args.labels:
                execute_soft_merge_by_label(driver, DATABASE, target_label=label)

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
# 5. ENTRY POINT & ARGUMENT PARSING
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
