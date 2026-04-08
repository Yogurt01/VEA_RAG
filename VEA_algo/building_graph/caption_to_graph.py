# building_graph/caption_to_graph.py
import argparse
import json
import os
import re
from pathlib import Path

import torch
from openai import OpenAI

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

RELATION_TO_IDX = {
    "BACKGROUND":  0,
    "CAUSE":       1,
    "COMPARISON":  2,
    "PREPARATION": 3,
    "RESULT":      4,
    "SUPPLEMENT":  5,
    "LIST":        6,
    "SUMMARY":     7,
    "RESTATEMENT": 8,
}

SYSTEM_PROMPT = """You are an expert in video discourse analysis based on Rhetorical Structure Theory (RST). Your task is to analyze a sequence of video scene captions and construct a single connected discourse graph representing how scenes relate to each other narratively and logically.

## TASK DESCRIPTION

Given a list of scenes from a video, each with a short caption, you must output a directed discourse graph where:
- Each NODE is a scene (Scene_0, Scene_1, ...).
- Each EDGE is a directed relation indicating how one scene modifies, supports, or depends on another.
- The graph must be FULLY CONNECTED: every node must be reachable through the edges. There must be NO isolated nodes and NO disconnected subgraphs.
- Each EDGE is labeled with exactly one of the nine rhetorical relations defined below.

## CONNECTIVITY REQUIREMENT

This is the most important structural constraint:
- Every scene must participate in at least one edge.
- The entire set of scenes must form one single connected component.
- After constructing all edges, verify that no scene is left isolated or separated from the rest.
- If a scene appears isolated, add the most appropriate edge connecting it to the main graph.

## ANTI-PATTERN: DO NOT CREATE LINEAR CHAINS

A linear chain where every scene connects only to the next scene (S0->S1->S2->...->SN) 
is WRONG. This is just a timeline, not a discourse graph.

A correct discourse graph has:
- One or more CENTRAL (nuclear) scenes that receive multiple incoming edges from different scenes.
- BRANCHING structure: multiple scenes pointing to the same target.
- HIERARCHICAL structure: some scenes group under a local nucleus before that nucleus connects to the global nucleus.
- Long-distance edges: Scene_2 can connect directly to Scene_15 if Scene_2 provides the background for Scene_15, skipping intermediate scenes.

Ask yourself for each scene: does this scene DIRECTLY support, cause, or 
relate to the IMMEDIATELY next scene — or does it actually relate to a 
LATER, more central scene? If the latter, draw the edge to that later scene.

## IDENTIFY TOPIC GROUPS FIRST

Before drawing any edges:
1. Identify distinct topic groups within the video.
2. Find the most nuclear scene within each group.
3. Find the most nuclear scene across all groups (global nucleus).
4. Connect within-group scenes to their local nucleus first.
5. Connect local nuclei to the global nucleus.
6. Add cross-group edges where one group supplements, causes, or provides background for another.
   
## RHETORICAL RELATION DEFINITIONS

1. BACKGROUND
   A scene provides contextual or situational background for another scene.
   Direction: background scene -> main scene it contextualizes

2. CAUSE
   A scene describes an event or action that causes an unexpected or consequential event in another scene.
   Direction: cause scene -> consequence scene

3. COMPARISON
   Two or more scenes present contrasting or parallel information at equal significance.
   Direction: each compared scene -> the other (mutual), or both -> shared parent

4. PREPARATION
   A scene describes a procedural step or setup that leads directly into another scene.
   Direction: preparatory scene -> subsequent scene

5. RESULT
   A scene shows the natural outcome or consequence of another scene, without strong causality.
   Direction: preceding scene -> result scene

6. SUPPLEMENT
   A scene adds supplementary or emphasizing information to another scene, without being essential.
   Direction: supplementing scene -> scene it supplements

7. LIST
   Two or more scenes are of equal narrative importance, occurring simultaneously or in parallel.
   Direction: each listed scene -> their shared parent scene

8. SUMMARY
   A scene summarizes or represents the content of a larger span of scenes.
   Direction: summary scene -> scene being summarized

9. RESTATEMENT
   A scene repeats or replays the content of another scene (e.g., slow-motion replay).
   Direction: restatement scene -> original scene

## OUTPUT FORMAT

Return a JSON object with one field:
- "edges": a list of directed edges, each with "from", "to", and "relation".

Critical rules:
- Every scene identifier (Scene_0, Scene_1, ...) must appear in at least one edge.
- The graph must form a single connected component — verify this before outputting.
- Do not output isolated nodes or disconnected subgraphs.
"""

USER_PROMPT_TEMPLATE = """Below are two examples drawn from real video discourse analysis. Study them carefully before constructing the graph for your assigned scenes.

---
## EXAMPLE 1 — High Jump Competition (sports video)

This video covers a high jump competition featuring three competitors. The third competitor wins. The second competitor joins the lap of honor.

Input scenes:

Scene_0: The audience gathers and the competition arena is shown, establishing the setting of a high jump event.
Scene_1: The first competitor approaches the bar and makes his jump attempt, clearing a lower height.
Scene_2: A close-up of the scoreboard shows the current standings after the first competitor's attempt.
Scene_3: The second competitor approaches and makes his jump attempt at a higher height, landing successfully.
Scene_4: The third competitor approaches and makes his jump attempt at the highest height, landing successfully to win the competition.
Scene_5: The audience applauds and the second and third competitors join the lap of honor together.
Scene_6: The final scoreboard is shown, confirming the third competitor as the winner of the event.

Expected output:

{{
  "edges": [
    {{"from": "Scene_0", "to": "Scene_4", "relation": "BACKGROUND"}},
    {{"from": "Scene_1", "to": "Scene_3", "relation": "COMPARISON"}},
    {{"from": "Scene_3", "to": "Scene_4", "relation": "COMPARISON"}},
    {{"from": "Scene_2", "to": "Scene_4", "relation": "SUPPLEMENT"}},
    {{"from": "Scene_5", "to": "Scene_4", "relation": "RESULT"}},
    {{"from": "Scene_6", "to": "Scene_4", "relation": "SUMMARY"}}
  ]
}}

Explanation of connectivity:
- Scene_4 (the third competitor wins) is the most nuclear scene — the central event of the video.
- Scene_0 provides background for Scene_4.
- Scene_1 and Scene_3 are compared with each other as preceding attempts, and Scene_3 is also compared with Scene_4.
- Scene_2 supplements Scene_4 with factual score context.
- Scene_5 results from Scene_4 (the winner's lap of honor).
- Scene_6 summarizes Scene_4 (the final confirmation).
- Every scene is connected. No isolated nodes exist.

---
## EXAMPLE 2 — Boris Johnson Visits Police Trainees (news video)

This video shows British Prime Minister Boris Johnson delivering a speech at a police training center. A trainee falls ill during the speech. Johnson returns to check on her after leaving the rostrum.

Input scenes:

Scene_0: At a police training center, Boris Johnson delivers a speech in front of a rostrum, establishing the formal public event.
Scene_1: While Johnson speaks, a police trainee standing behind him sits down due to illness, introducing an unexpected disruption.
Scene_2: Johnson concludes his speech by thanking the police trainees, marking the end of his formal address.
Scene_3: Johnson walks away from the rostrum, transitioning out of the official setting.
Scene_4: Johnson immediately returns to the trainee who sat down, responding directly to the disruption caused by her illness.
Scene_5: Johnson speaks personally with the ill trainee, engaging in direct one-on-one interaction.
Scene_6: Johnson then speaks with a group of other trainees, extending his personal engagement to the broader group.
Scene_7: Johnson walks away from the trainees, closing the interpersonal exchange and ending the event.

Expected output:

{{
  "edges": [
    {{"from": "Scene_0", "to": "Scene_2", "relation": "BACKGROUND"}},
    {{"from": "Scene_1", "to": "Scene_4", "relation": "CAUSE"}},
    {{"from": "Scene_2", "to": "Scene_6", "relation": "RESULT"}},
    {{"from": "Scene_3", "to": "Scene_4", "relation": "PREPARATION"}},
    {{"from": "Scene_4", "to": "Scene_5", "relation": "PREPARATION"}},
    {{"from": "Scene_5", "to": "Scene_6", "relation": "LIST"}},
    {{"from": "Scene_7", "to": "Scene_6", "relation": "SUPPLEMENT"}}
  ]
}}

Explanation of connectivity:
- Scene_6 (Johnson talking with multiple trainees) is the most nuclear scene — it is the culminating interpersonal event.
- Scene_0 contextualizes Scene_2 (the speech conclusion).
- Scene_1 directly causes Scene_4 (the illness prompts Johnson to return).
- Scene_2 results in Scene_6 (the public speech transitions into personal interaction).
- Scene_3 prepares for Scene_4 (walking away sets up the contrast of returning).
- Scene_4 prepares for Scene_5 (returning enables the personal conversation).
- Scene_5 and Scene_6 are listed as sequential parallel interactions of equal significance.
- Scene_7 supplements Scene_6 as a closing action.
- Every scene is connected into a single graph. No isolated nodes.

---
## CONNECTIVITY CHECKLIST

Before finalizing your output, verify:
1. List all scene identifiers: Scene_0, Scene_1, ..., Scene_N.
2. Check that each scene appears in at least one edge (as "from" or "to").
3. Confirm that all scenes are reachable through the edges — no disconnected subgraphs.
4. If any scene is missing from all edges, add the most appropriate edge to connect it.

---
## YOUR TASK

Now analyze the following scenes and produce the discourse graph. Apply the connectivity checklist before outputting.

Input scenes:

{scenes_text}

Output:"""


class GraphBuilder:
    """
    Build discourse graphs from scene captions using an LLM,
    then persist edge_index and edge_attr into the existing scene_embeddings.pt.
    """

    def __init__(self, args):
        self.root_dir = Path(args.root_dir)
        self.segmentation_filename = "segments.json"
        self.embedding_filename = "scene_embeddings.pt"
        self.label_file = self.root_dir / "video_ids_label.json"

        all_video_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        limit = getattr(args, "limit_videos", None)
        self.video_dirs = all_video_dirs[:limit] if limit and limit > 0 else all_video_dirs

        self.videos = []
        for vdir in self.video_dirs:
            mp4_files = list(vdir.glob("*.mp4"))
            if mp4_files:
                self.videos.append(mp4_files[0])

        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        self.filename_to_label = self._load_labels()

        self.force = args.force

    def _load_labels(self) -> dict:
        """
        Load video labels from video_ids_label.json.
        Returns a dict mapping video filename (stem) to integer label.
        """
        if not self.label_file.exists():
            print(f"Warning: label file not found at {self.label_file}")
            return {}

        with open(self.label_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        filename_to_label = {}
        for label_str, filenames in raw.items():
            label_int = int(label_str)
            for filename in filenames:
                stem = Path(filename).stem
                filename_to_label[stem] = label_int

        return filename_to_label

    def _call_llm(self, scenes_text: str) -> dict:
        """
        Call the LLM to generate a discourse graph from scene captions.
        Returns parsed dict with "edges" key, or empty dict on failure.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(scenes_text=scenes_text)

        response = self.openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
        )

        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"Failed to parse LLM response: {raw[:120]}")
                return {}

    def _parse_scene_id(self, scene_str: str) -> int:
        """
        Extract integer index from scene identifier string.
        Accepts formats: "Scene_0", "Scene_1", "0", "1".
        """
        match = re.search(r"\d+", str(scene_str))
        if match:
            return int(match.group())
        raise ValueError(f"Cannot parse scene id from: {scene_str}")

    def _edges_to_pyg_format(
        self,
        edges: list,
        num_nodes: int,
    ) -> tuple:
        """
        Convert LLM edge list to PyG-compatible tensors.

        Returns:
            edge_index : LongTensor of shape [2, num_edges]
                         edge_index[0] = source nodes
                         edge_index[1] = target nodes
            edge_attr  : LongTensor of shape [num_edges]
                         integer-encoded relation type per edge
        """
        src_list      = []
        dst_list      = []
        edge_type_list = []

        for edge in edges:
            try:
                src      = self._parse_scene_id(edge["from"])
                dst      = self._parse_scene_id(edge["to"])
                relation = edge.get("relation", "").upper()
            except (KeyError, ValueError) as e:
                print(f"Skipping malformed edge {edge}: {e}")
                continue

            if src >= num_nodes or dst >= num_nodes:
                print(f"Skipping out-of-range edge: {src} -> {dst} (num_nodes={num_nodes})")
                continue

            relation_idx = RELATION_TO_IDX.get(relation, -1)
            if relation_idx == -1:
                print(f"Unknown relation '{relation}', skipping edge {src} -> {dst}")
                continue

            src_list.append(src)
            dst_list.append(dst)
            edge_type_list.append(relation_idx)

        if not src_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0,),   dtype=torch.long)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr  = torch.tensor(edge_type_list,       dtype=torch.long)

        return edge_index, edge_attr

    def _caption_to_graph(self):
        """
        For each video:
          1. Load captions from segments.json.
          2. Call LLM to produce discourse edges.
          3. Convert edges to PyG edge_index and edge_attr.
          4. Load scene_embeddings.pt and add graph fields in-place.
          5. Save back to scene_embeddings.pt.

        PyG Data object fields after this step:
          x          : FloatTensor [num_nodes, 2048]  — node features (embeddings)
          edge_index : LongTensor  [2, num_edges]     — COO adjacency
          edge_attr  : LongTensor  [num_edges]        — relation type index
          y          : LongTensor  [1]                — video-level engagement label

        Note on label handling:
          y is stored per graph (video-level), not per node.
          When building a PyG dataset for GNN training, split video-level graphs
          into train/val/test externally — do NOT split individual nodes.
          Use torch_geometric.data.Data(x=..., edge_index=..., edge_attr=..., y=...)
          and wrap in an InMemoryDataset or DataLoader.

        Relation index mapping (stored in edge_attr):
          0: BACKGROUND   1: CAUSE       2: COMPARISON
          3: PREPARATION  4: RESULT      5: SUPPLEMENT
          6: LIST         7: SUMMARY     8: RESTATEMENT
        """
        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file   = video_dir / self.segmentation_filename
            embedding_file = video_dir / self.embedding_filename

            if not segment_file.exists():
                print(f"No segments.json in {video_dir.name}, skipping.")
                continue
            if not embedding_file.exists():
                print(f"No scene_embeddings.pt in {video_dir.name}, skipping.")
                continue

            with open(segment_file, "r", encoding="utf-8") as f:
                segments = json.load(f)

            # Check all scenes have captions before proceeding
            missing = [i for i, s in enumerate(segments) if not s.get("caption", "").strip()]
            if missing:
                print(f"Skipping {video_dir.name}: scenes {missing} have no caption.")
                continue

            checkpoint = torch.load(embedding_file, map_location="cpu")

            if not self.force:
                if "edge_index" in checkpoint and "edge_attr" in checkpoint:
                    print(f"Skipping {video_dir.name}: graph already exists. Use force=True to overwrite.")
                    continue
            else:
                print(f"Force re-building graph for {video_dir.name}...")

            num_nodes = checkpoint["embeddings"].shape[0]

            # Build scenes_text for the LLM prompt
            scenes_text = "\n".join(
                f"Scene_{i}: {seg['caption']}"
                for i, seg in enumerate(segments)
                if i < num_nodes
            )

            print(f"Building graph for {video_dir.name} ({num_nodes} nodes)...")
            graph_output = self._call_llm(scenes_text)

            edges = graph_output.get("edges", [])
            if not edges:
                print(f"Warning: no edges returned for {video_dir.name}.")

            edge_index, edge_attr = self._edges_to_pyg_format(edges, num_nodes)

            # Resolve video-level label
            video_stem = video_path.stem
            label      = self.filename_to_label.get(video_stem, -1)
            if label == -1:
                print(f"Warning: no label found for {video_path.name}, storing y=-1.")
            y = torch.tensor([label], dtype=torch.long)

            # Persist graph data into the existing checkpoint
            checkpoint["edge_index"] = edge_index
            checkpoint["edge_attr"] = edge_attr
            checkpoint["y"] = y
            checkpoint["relation_map"] = RELATION_TO_IDX  # for reference

            torch.save(checkpoint, embedding_file)
            print(
                f"Saved: {video_dir.name} | "
                f"nodes={num_nodes} | "
                f"edges={edge_index.shape[1]} | "
                f"label={label}"
            )

    def run(self):
        self._caption_to_graph()


def main():
    parser = argparse.ArgumentParser(description="Build discourse graphs from scene captions.")
    parser.add_argument("--root_dir",     type=str, required=True,
                        help="Root dataset directory containing video folders.")
    parser.add_argument("--limit_videos", type=int, default=0,
                        help="Process only the first N videos (0 = all).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing graphs.")
    args = parser.parse_args()

    builder = GraphBuilder(args)
    builder.run()


if __name__ == "__main__":
    main()