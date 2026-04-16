#!/usr/bin/env python3
"""Build RST graph for each video folder from:
1) segments.json (captions)
2) rst_tree.tree (LLM-generated bottom-up tree)
3) scene_embeddings.pt (to write graph fields back)

Pipeline per video:
- captions -> captions.txt + captions.edus.txt
- CoreNLP -> captions.txt.xml -> captions.txt.conll
- map EDU indices -> captions.txt.merge
- convert rst_tree.tree -> captions.txt.brackets
- rstscr logic (merge + brackets) -> rst_links
- rst_links -> edge_index / edge_attr
- write edge_index, edge_attr, rst_links, y into scene_embeddings.pt
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import torch

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPLP.map_edus_to_merge import align_edus_to_conll, read_conll, read_edu_lines, write_merge
from DPLP.preprocess.xmlreader_upto_generating_conll_files import combine, reader, writer
from DPLP.rstscr import new_read_bracket, read_discourse_merge


EDGE_TYPES = [
    "elaboration",
    "span",
    "list",
    "same_unit",
    "textualorganization",
    "topic",
    "ROOT",
    "attribution",
    "contrast",
    "circumstance",
    "purpose",
    "temporal",
    "explanation",
    "means",
    "reason",
    "condition",
    "concession",
    "antithesis",
    "example",
    "sequence",
    "manner",
    "question",
    "comparison",
    "disjunction",
    "result",
    "summary",
    "comment",
    "definition",
    "evidence",
    "restatement",
    "evaluation",
    "consequence",
    "hypothetical",
    "rhetorical",
    "interpretation",
    "background",
    "inverted",
    "enablement",
    "contingency",
    "cause",
    "statement",
    "preference",
    "analogy",
]
DEFAULT_ETYPE_MAP = {k: i for i, k in enumerate(EDGE_TYPES)}
DEFAULT_UNKNOWN_REL = "span"


@dataclass
class LLMTreeNode:
    label: str
    children: List["LLMTreeNode"] = field(default_factory=list)
    span: Tuple[int, int] | None = None
    role: str = "Nucleus"
    relation: str = "span"


TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
LEAF_RE = re.compile(r"^text$", flags=re.IGNORECASE)
IDX_RE = re.compile(r"^\d+$")


def parse_llm_tree(tree_text: str) -> LLMTreeNode:
    tokens = TOKEN_RE.findall(tree_text)
    idx = 0

    def parse_node() -> LLMTreeNode:
        nonlocal idx
        if idx >= len(tokens) or tokens[idx] != "(":
            raise ValueError("Expected '(' while parsing rst_tree.tree")
        idx += 1

        if idx >= len(tokens):
            raise ValueError("Unexpected end of tree")
        label = tokens[idx]
        idx += 1

        if LEAF_RE.match(label):
            if idx >= len(tokens) or not IDX_RE.match(tokens[idx]):
                raise ValueError("Leaf must be '(text <int>)'")
            leaf_idx = tokens[idx]
            idx += 1
            if idx >= len(tokens) or tokens[idx] != ")":
                raise ValueError("Missing ')' after leaf")
            idx += 1
            return LLMTreeNode(label=f"text {leaf_idx}")

        node = LLMTreeNode(label=label)
        while idx < len(tokens) and tokens[idx] == "(":
            node.children.append(parse_node())
        if idx >= len(tokens) or tokens[idx] != ")":
            raise ValueError(f"Missing ')' for node: {label}")
        idx += 1
        return node

    root = parse_node()
    if idx != len(tokens):
        raise ValueError("Trailing tokens after tree parse")
    return root


def split_pattern_relation(label: str) -> Tuple[str, str]:
    if ":" in label:
        p, r = label.split(":", 1)
    else:
        p, r = label, "span"
    return p.lower(), r.lower()


def assign_spans(node: LLMTreeNode) -> None:
    if node.label.startswith("text "):
        zidx = int(node.label.split()[1])
        one_based = zidx + 1
        node.span = (one_based, one_based)
        return
    for ch in node.children:
        assign_spans(ch)
    node.span = (node.children[0].span[0], node.children[-1].span[1])


def assign_roles_relations(node: LLMTreeNode) -> None:
    if node.label.startswith("text "):
        return
    pattern, rel = split_pattern_relation(node.label)
    if len(node.children) == 0:
        return

    if len(node.children) == 1:
        roles = ["Nucleus"]
    elif pattern == "nucleus-nucleus":
        roles = ["Nucleus", "Nucleus"]
    elif pattern == "nucleus-satellite":
        roles = ["Nucleus", "Satellite"]
    elif pattern == "satellite-nucleus":
        roles = ["Satellite", "Nucleus"]
    elif pattern == "satellite":
        roles = ["Satellite"] * len(node.children)
    else:
        roles = ["Nucleus"] * len(node.children)

    if len(roles) < len(node.children):
        roles.extend(["Nucleus"] * (len(node.children) - len(roles)))

    for i, ch in enumerate(node.children):
        role = roles[i]
        ch.role = role
        if pattern in {"nucleus-satellite", "satellite-nucleus"}:
            ch.relation = "span" if role == "Nucleus" else rel
        elif pattern == "nucleus-nucleus":
            ch.relation = rel
        else:
            ch.relation = rel if role == "Satellite" else "span"
        assign_roles_relations(ch)


def collect_postorder(node: LLMTreeNode, out: List[Tuple[Tuple[int, int], str, str]]) -> None:
    for ch in node.children:
        collect_postorder(ch, out)
    out.append((node.span, node.role, node.relation))


def convert_llm_tree_to_brackets(tree_text: str) -> List[Tuple[Tuple[int, int], str, str]]:
    root = parse_llm_tree(tree_text)
    root.role = "Nucleus"
    root.relation = "ROOT"
    assign_spans(root)
    assign_roles_relations(root)
    spans: List[Tuple[Tuple[int, int], str, str]] = []
    collect_postorder(root, spans)
    return spans


def write_brackets_file(spans: List[Tuple[Tuple[int, int], str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for span, nuc, rel in spans:
            f.write(f"({span}, '{nuc}', '{rel}')\n")


def normalize_caption_text(text: str) -> str:
    text = (text or "").strip()
    # Put spaces around punctuation for whitespace tokenizer mode.
    text = text.replace('—', ", ")
    text = re.sub(r'([,.;:!?()"\'/\\-])', r" \1 ", text)
    # Normalize spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


class RSTCaptionGraphBuilder:
    def __init__(
        self,
        videos_root: Path,
        segmentation_filename: str = "segments.json",
        embedding_filename: str = "scene_embeddings.pt",
        rst_tree_filename: str = "rst_tree.tree",
        filename_to_label: Dict[str, int] | None = None,
        force: bool = False,
        java_cmd: str = "java",
    ):
        self.videos_root = Path(videos_root).resolve()
        self.segmentation_filename = segmentation_filename
        self.embedding_filename = embedding_filename
        self.rst_tree_filename = rst_tree_filename
        self.filename_to_label = filename_to_label or {}
        self.force = force
        self.java_cmd = java_cmd
        self.repo_root = REPO_ROOT

    def _video_dirs(self) -> Iterable[Path]:
        for p in sorted(self.videos_root.iterdir()):
            if p.is_dir():
                yield p

    def _run(self, cmd: List[str], cwd: Path) -> None:
        subprocess.run(cmd, cwd=str(cwd), check=True)

    def _resolve_label(self, video_dir: Path) -> int:
        if video_dir.name in self.filename_to_label:
            return int(self.filename_to_label[video_dir.name])
        mp4s = sorted(video_dir.glob("*.mp4"))
        for mp4 in mp4s:
            if mp4.name in self.filename_to_label:
                return int(self.filename_to_label[mp4.name])
        for mp4 in mp4s:
            if mp4.stem in self.filename_to_label:
                return int(self.filename_to_label[mp4.stem])
        return -1

    def _run_corenlp(self, work_dir: Path, text_file: Path) -> Path:
        classpath = str(self.repo_root / "DPLP" /"*")
        cmd = [
            self.java_cmd,
            "-mx2g",
            "-cp",
            classpath,
            "edu.stanford.nlp.pipeline.StanfordCoreNLP",
            "-annotators",
            "tokenize,ssplit,pos,lemma,ner,parse",
            "-ssplit.eolonly",
            "true",
            "-tokenize.whitespace",
            "true",
            "-file",
            str(text_file.resolve()),
            "-outputFormat",
            "xml",
        ]
        self._run(cmd, work_dir)
        xml_file = text_file.resolve().with_suffix(text_file.suffix + ".xml")
        if not xml_file.exists():
            raise FileNotFoundError(f"CoreNLP output not found: {xml_file}")
        return xml_file

    def _xml_to_conll(self, xml_file: Path) -> Path:
        conll_file = xml_file.with_suffix(".conll")
        sentlist, constlist = reader(str(xml_file))
        sentlist = combine(sentlist, constlist)
        writer(sentlist, str(conll_file))
        return conll_file

    def _build_merge(self, conll_file: Path, edu_file: Path) -> Path:
        rows = read_conll(str(conll_file))
        edus = read_edu_lines(str(edu_file))
        edu_idx_per_token = align_edus_to_conll(rows, edus)
        merge_file = conll_file.with_suffix(".merge")
        write_merge(rows, edu_idx_per_token, str(merge_file))
        return merge_file

    def _resolve_tree_file(self, video_dir: Path) -> Path | None:
        p1 = video_dir / self.rst_tree_filename
        if p1.exists():
            return p1
        p2 = video_dir / "video_caps.rst_tree.tree"
        if p2.exists():
            return p2
        return None

    def _build_brackets_from_llm_tree(
        self, tree_file: Path, bracket_file: Path
    ) -> List[Tuple[Tuple[int, int], str, str]]:
        tree_text = tree_file.read_text(encoding="utf-8")
        spans = convert_llm_tree_to_brackets(tree_text)
        write_brackets_file(spans, bracket_file)
        return spans

    def _rst_links(self, merge_file: Path, bracket_file: Path) -> List[Tuple[int, int, str]]:
        if not merge_file.exists() or not bracket_file.exists():
            return []
        _, edu_pool, edu_nsubj = read_discourse_merge(str(merge_file))
        _, links = new_read_bracket(str(bracket_file), edu_pool, edu_nsubj)
        return links

    def _relation_to_idx(self, rel: str) -> int:
        if rel in DEFAULT_ETYPE_MAP:
            return DEFAULT_ETYPE_MAP[rel]
        rel_l = rel.lower()
        if rel_l in DEFAULT_ETYPE_MAP:
            return DEFAULT_ETYPE_MAP[rel_l]
        return DEFAULT_ETYPE_MAP[DEFAULT_UNKNOWN_REL]

    def _links_to_pyg(
        self, links: List[Tuple[int, int, str]], num_nodes: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if torch is None:
            raise ImportError("torch is required. Please install PyTorch.")
        edge_pairs: List[List[int]] = []
        edge_types: List[int] = []
        for src_edu, dst_edu, rel in links:
            src = int(src_edu) - 1
            dst = int(dst_edu) - 1
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                edge_pairs.append([src, dst])
                edge_types.append(self._relation_to_idx(str(rel)))

        if not edge_pairs:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
        return edge_index, edge_attr

    def _process_one_video(self, video_dir: Path) -> None:
        if torch is None:
            raise ImportError("torch is required. Please install PyTorch.")

        segment_file = video_dir / self.segmentation_filename
        embedding_file = video_dir / self.embedding_filename
        tree_file = self._resolve_tree_file(video_dir)

        if not segment_file.exists():
            print(f"No {self.segmentation_filename} in {video_dir.name}, skipping.")
            return
        if not embedding_file.exists():
            print(f"No {self.embedding_filename} in {video_dir.name}, skipping.")
            return
        if tree_file is None:
            print(f"No rst tree file in {video_dir.name}, expected {self.rst_tree_filename} or video_caps.rst_tree.tree")
            return

        ckpt = torch.load(embedding_file, map_location="cpu")
        if "embeddings" not in ckpt:
            print(f"Skipping {video_dir.name}: missing embeddings in checkpoint.")
            return

        if not self.force and ("edge_index" in ckpt and "edge_attr" in ckpt and "rst_links" in ckpt):
            print(f"Skipping {video_dir.name}: graph exists (use --force to overwrite).")
            return

        with open(segment_file, "r", encoding="utf-8") as f:
            segments = json.load(f)

        num_nodes = int(ckpt["embeddings"].shape[0])
        captions = []
        for i, seg in enumerate(segments):
            if i >= num_nodes:
                break
            captions.append(normalize_caption_text(seg.get("caption") or ""))

        if len(captions) < num_nodes:
            print(f"Skipping {video_dir.name}: captions < num_nodes ({len(captions)} < {num_nodes}).")
            return
        missing = [i for i, c in enumerate(captions) if not c]
        if missing:
            print(f"Skipping {video_dir.name}: empty captions at scenes {missing}.")
            return

        print(f"Building graph for {video_dir.name} ({num_nodes} nodes)...")
        work_dir = video_dir / "rst_work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        text_file = work_dir / "captions.txt"
        edu_file = work_dir / "captions.edus.txt"
        text_file.write_text("\n".join(captions) + "\n", encoding="utf-8")
        shutil.copyfile(text_file, edu_file)

        xml_file = self._run_corenlp(work_dir, text_file)
        conll_file = self._xml_to_conll(xml_file)
        merge_file = self._build_merge(conll_file, edu_file)

        bracket_file = merge_file.with_suffix(".brackets")
        spans = self._build_brackets_from_llm_tree(tree_file, bracket_file)
        max_edu_in_tree = max(span[0][1] for span in spans) if spans else 0
        if max_edu_in_tree > num_nodes:
            raise ValueError(
                f"rst_tree.tree uses EDU index up to {max_edu_in_tree}, "
                f"but scene_embeddings has only {num_nodes} nodes."
            )

        links = self._rst_links(merge_file, bracket_file)
        edge_index, edge_attr = self._links_to_pyg(links, num_nodes)
        label = self._resolve_label(video_dir)
        y = torch.tensor([label], dtype=torch.long)

        ckpt["edge_index"] = edge_index
        ckpt["edge_attr"] = edge_attr
        ckpt["rst_links"] = links
        ckpt["y"] = y
        torch.save(ckpt, embedding_file)

        print(
            f"Saved: {video_dir.name} | nodes={num_nodes} | "
            f"edges={edge_index.shape[1]} | label={label}"
        )

    def run(self) -> None:
        for video_dir in self._video_dirs():
            try:
                self._process_one_video(video_dir)
            except Exception as exc:
                print(f"Error in {video_dir.name}: {exc}")


def load_label_map(path: Path | None) -> Dict[str, int]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Accept:
    # A) {"video_07": 1, "video_09": 0}
    # B) {"1": ["joey-find-out", ...], "0": ["how-i-met-your-mother", ...]}
    if all(isinstance(v, list) for v in raw.values()):
        out: Dict[str, int] = {}
        for k, names in raw.items():
            label = int(k)
            for name in names:
                s = str(name)
                out[s] = label
                if s.endswith(".mp4"):
                    out[s[:-4]] = label
        return out
    return {str(k): int(v) for k, v in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build RST graph from segments.json + rst_tree.tree and save to scene_embeddings.pt"
    )
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--segmentation-filename", default="segments.json")
    parser.add_argument("--embedding-filename", default="scene_embeddings.pt")
    parser.add_argument("--rst-tree-filename", default="rst_tree.tree")
    parser.add_argument("--label-map-json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--java-cmd", default="java")
    args = parser.parse_args()

    label_map = load_label_map(Path(args.label_map_json).resolve()) if args.label_map_json else {}

    runner = RSTCaptionGraphBuilder(
        videos_root=Path(args.videos_root),
        segmentation_filename=args.segmentation_filename,
        embedding_filename=args.embedding_filename,
        rst_tree_filename=args.rst_tree_filename,
        filename_to_label=label_map,
        force=args.force,
        java_cmd=args.java_cmd,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
