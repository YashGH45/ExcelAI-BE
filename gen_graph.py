"""
Generates graph.json — a lightweight map of the project for token-efficient navigation.
Run: python gen_graph.py
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).parent
EXCLUDE = {"venv", "__pycache__", ".git", "uploads", "vector_db"}

def parse_file(path: Path) -> dict:
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    info = {"imports": [], "classes": {}, "functions": []}

    for node in ast.walk(tree):
        # Top-level imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names]
            info["imports"].append(f"{mod}:{','.join(names)}")

        # Top-level functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node.parent if hasattr(node, "parent") else None, ast.Module):
                info["functions"].append(node.name)

        # Classes and their methods
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in ast.walk(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            info["classes"][node.name] = methods

    # Re-extract top-level functions properly
    info["functions"] = [
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    return info


def local_dep(imports: list[str], file_names: set[str]) -> list[str]:
    """Filter imports that match local project files."""
    deps = []
    for imp in imports:
        mod = imp.split(":")[0].split(".")[0]
        if f"{mod}.py" in file_names:
            deps.append(mod)
    return list(set(deps))


def build_graph():
    py_files = [
        p for p in ROOT.rglob("*.py")
        if not any(ex in p.parts for ex in EXCLUDE)
    ]

    file_names = {p.name for p in py_files}
    graph = {}

    for path in sorted(py_files):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        data = parse_file(path)
        if not data:
            continue
        graph[rel] = {
            "classes": data["classes"],
            "functions": data["functions"],
            "local_deps": local_dep(data["imports"], file_names),
            "external_imports": [
                i.split(":")[0] for i in data["imports"]
                if i.split(":")[0].split(".")[0] + ".py" not in file_names
            ][:10],  # cap to 10 to keep graph small
        }

    out = ROOT / "graph.json"
    out.write_text(json.dumps(graph, indent=2))
    print(f"graph.json written — {len(graph)} files mapped")
    return graph


if __name__ == "__main__":
    build_graph()