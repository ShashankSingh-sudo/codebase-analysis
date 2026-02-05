"""
Code Analyzer - Streamlit App
Clones a Git repo, explores structure, and uses Hugging Face LLM for code analysis.
"""

import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from pathlib import Path

# Code file extensions to analyze
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".scala", ".r", ".sql"
}

# Hugging Face models (router API - use models that work with your enabled providers)
HF_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]

# 5 fixed analysis parameters - all code analyzed against these
ANALYSIS_PARAMS = [
    ("functions", "Functions & Methods", "🔧"),
    ("endpoints", "Endpoints / APIs", "🌐"),
    ("dependencies", "Dependencies", "📦"),
    ("optimizations", "Optimizations", "⚡"),
    ("issues", "Potential Issues", "⚠️"),
]


def normalize_repo_url(url: str) -> str:
    """Convert GitHub/GitLab web URLs to valid git clone URLs."""
    url = url.strip()
    # Remove trailing slashes
    url = url.rstrip("/")
    # GitHub: .../owner/repo/tree/branch or /blob/... or just .../owner/repo
    if "github.com" in url:
        if "/tree/" in url or "/blob/" in url:
            url = url.split("/tree/")[0].split("/blob/")[0]
        if not url.endswith(".git"):
            url = url + ".git"
    # GitLab: .../owner/repo/-/tree/branch
    elif "gitlab.com" in url:
        if "/-/tree/" in url or "/-/blob/" in url:
            url = url.split("/-/tree/")[0].split("/-/blob/")[0]
        if not url.endswith(".git"):
            url = url + ".git"
    return url


def clone_repo(repo_url: str, target_dir: str, sparse_path: str | None = None) -> tuple[bool, str]:
    """Clone a git repository. If sparse_path given, clone only that folder (sparse checkout)."""
    import shutil
    import subprocess
    from git import Repo
    clone_url = normalize_repo_url(repo_url)
    try:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        if sparse_path and sparse_path.strip():
            path = sparse_path.strip().strip("/")
            try:
                r = subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", clone_url, target_dir], capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr or r.stdout)
                r2 = subprocess.run(["git", "sparse-checkout", "set", path], cwd=target_dir, capture_output=True, text=True)
                if r2.returncode != 0:
                    raise RuntimeError(r2.stderr or "Path may not exist in repo")
                return True, f"Cloned (sparse): /{path}"
            except (FileNotFoundError, RuntimeError):
                shutil.rmtree(target_dir, ignore_errors=True)
                Repo.clone_from(clone_url, target_dir, depth=1)
                return True, "Full clone (sparse failed)"
        Repo.clone_from(clone_url, target_dir, depth=1)
        return True, "Repository cloned successfully!"
    except Exception as e:
        return False, str(e)


def get_folder_structure(root_path: str) -> list[str]:
    """Get all folders in the repo (excluding .git and common ignores)."""
    folders = set()
    ignore = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
    
    for dirpath, dirnames, _ in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in ignore and not d.startswith(".")]
        rel_path = os.path.relpath(dirpath, root_path)
        if rel_path != ".":
            folders.add(rel_path.replace("\\", "/"))
    folders.add(".")
    return sorted(folders, key=lambda x: (x.count("/") + x.count("\\"), x))


def extract_functions_from_code(filepath: str, content: str) -> list[dict]:
    """Extract function/class/method names from code using regex."""
    items = []
    ext = Path(filepath).suffix.lower()
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Python: def, class, async def
        if ext == ".py":
            m = re.match(r"^(async\s+)?def\s+(\w+)\s*\(", stripped)
            if m:
                items.append({"type": "function", "name": m.group(2), "line": i})
                continue
            m = re.match(r"^class\s+(\w+)", stripped)
            if m:
                items.append({"type": "class", "name": m.group(1), "line": i})
        # JS/TS: function, class
        elif ext in {".js", ".ts", ".jsx", ".tsx"}:
            m1 = re.search(r"function\s+(\w+)\s*\(", stripped)
            if m1:
                items.append({"type": "function", "name": m1.group(1), "line": i})
            m2 = re.search(r"^class\s+(\w+)", stripped)
            if m2:
                items.append({"type": "class", "name": m2.group(1), "line": i})
    return items


def get_code_files(folder_path: str) -> list[tuple[str, str]]:
    """Get all code files in folder and subfolders with their content."""
    files_content = []
    
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in CODE_EXTENSIONS:
                full_path = os.path.join(dirpath, fname)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    rel_path = os.path.relpath(full_path, folder_path).replace("\\", "/")
                    files_content.append((rel_path, content))
                except Exception:
                    pass
    return files_content


def get_hf_token() -> str:
    """Return HF token - from Space secrets, env, or fallback."""
    try:
        t = st.secrets.get("HF_TOKEN", "")
        if t:
            return t
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "")


def build_analysis_prompt(files_content: list[tuple[str, str]]) -> str:
    """Build the prompt for LLM code analysis."""
    combined = ""
    for rel_path, content in files_content:
        combined += f"\n\n--- File: {rel_path} ---\n{content[:3500]}"
    return combined


def split_into_batches(files: list, n_batches: int = 4) -> list[list]:
    """Split files into roughly equal batches for parallel processing."""
    n = min(n_batches, max(1, len(files)))
    batch_size = max(1, len(files) // n)
    return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]


def merge_analysis_results(results: list[dict]) -> dict:
    """Merge results from parallel LLM calls."""
    merged = {"functions": [], "endpoints": [], "dependencies": [], "optimizations": [], "issues": []}
    dep_seen = set()
    for data in results:
        if not data:
            continue
        for k in merged:
            items = data.get(k, [])
            if k == "dependencies":
                for item in items:
                    s = item if isinstance(item, str) else str(item)
                    if s and s not in dep_seen:
                        dep_seen.add(s)
                        merged[k].append(item)
            else:
                merged[k].extend(items)
    return merged


def parse_json_from_response(text: str) -> dict | None:
    """Extract and parse JSON from LLM response (handles markdown code blocks)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _analyze_batch(batch_files: list, model_id: str, hf_token: str) -> tuple[dict | None, dict, str | None]:
    """Analyze a single batch of files. Returns (parsed_data, usage_info, error_msg)."""
    combined = build_analysis_prompt(batch_files)
    success, result, usage = analyze_with_llm(combined, model_id, hf_token)
    if success:
        return parse_json_from_response(result), usage, None
    return None, usage, result


def analyze_with_llm(code_content: str, model_id: str, hf_token: str | None) -> tuple[bool, str, dict]:
    """Analyze code using Hugging Face router API (OpenAI-compatible). Returns (success, result, usage_info)."""
    import requests
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not hf_token:
        return False, "HF_TOKEN not set.", usage_info
    try:
        system_prompt = """You are an expert code analyst. Analyze ALL the provided code and return ONLY valid JSON (no markdown, no explanation).

Return ALL items you find — do NOT limit the number. Include every function, endpoint, dependency, optimization suggestion, and issue.

Structure (all 5 parameters required):
{
  "functions": [{"name":"str","description":"str","file":"str","parameters":["param1"],"return_type":"str or null","docstring":"str or null","line_start":int}],
  "endpoints": [{"path":"str","method":"GET|POST","description":"str"}],
  "dependencies": ["import1","import2"],
  "optimizations": [{"area":"performance|readability|security","suggestion":"str","location":"str"}],
  "issues": [{"type":"bug|antipattern|risk","description":"str","location":"str"}]
}

For functions: extract name, description, file, parameters list, return_type, docstring (first line), line_start. Be exhaustive — list every function/method.
Use empty arrays [] only when genuinely none found."""

        user_prompt = f"Analyze ALL code below and return complete JSON with every item found:\n{code_content}"

        r = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 4096,
                "temperature": 0.3,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            return False, "Model returned no choices.", usage_info
        msg = choices[0].get("message", {})
        result = msg.get("content", "")
        if not result:
            return False, "Model returned empty response.", usage_info
        usage = data.get("usage", {})
        if usage:
            usage_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        return True, result, usage_info
    except requests.exceptions.HTTPError as e:
        err = str(e)
        if e.response is not None:
            try:
                body = e.response.json()
                err = body.get("error", {}).get("message", err)
            except Exception:
                err = e.response.text[:300] if e.response.text else err
        if "401" in err or "unauthorized" in err.lower():
            err = f"Authentication failed. Check HF_TOKEN. {err}"
        elif "403" in err or "forbidden" in err.lower():
            err = f"Access denied. Model may require Pro or license. {err}"
        elif "model_not_supported" in err.lower():
            err = f"Model not available with your providers. Try another model. {err}"
        return False, err, usage_info
    except Exception as e:
        return False, str(e), usage_info


def main():
    st.set_page_config(page_title="Code Analyzer", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
        /* Enterprise-ready theme */
        :root { --primary: #0f4c81; --primary-light: #1e6bb8; --surface: #f8fafc; --border: #e2e8f0; --text-muted: #64748b; }
        .stApp { max-width: 1400px; margin: 0 auto; }
        
        /* Hero / Header */
        .hero {
            background: linear-gradient(135deg, #0f4c81 0%, #1a365d 50%, #0f172a 100%);
            color: white; padding: 2rem 2.5rem; border-radius: 12px;
            margin-bottom: 2rem; border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 20px rgba(15, 76, 129, 0.15);
        }
        .hero h1 { margin: 0; font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; }
        .hero p { margin: 0.6rem 0 0 0; opacity: 0.92; font-size: 0.95rem; line-height: 1.5; }
        
        /* Section containers */
        .section-title {
            font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;
            letter-spacing: 0.08em; margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* Analysis report sections */
        .report-section {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1.25rem 1.5rem; margin: 1rem 0;
            border-left: 4px solid #0f4c81;
        }
        .report-section h4 { margin: 0 0 0.75rem 0; font-size: 1rem; color: #0f172a; }
        .report-item {
            background: #fff; border: 1px solid #e2e8f0; border-radius: 6px;
            padding: 1rem 1.25rem; margin: 0.5rem 0;
            font-size: 0.9rem; line-height: 1.5;
        }
        .report-item-header { font-weight: 600; color: #0f4c81; margin-bottom: 0.25rem; }
        
        /* Metrics row */
        .metrics-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
        .metric-card {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1rem 1.5rem; min-width: 120px;
        }
        .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; }
        .metric-value { font-size: 1.25rem; font-weight: 600; color: #0f172a; }
        
        /* Footer */
        .footer { margin-top: 3rem; padding: 1.5rem; border-top: 1px solid #e2e8f0; font-size: 0.8rem; color: #94a3b8; text-align: center; }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Configuration")
        st.markdown("---")
        hf_token = get_hf_token()
        token_input = st.text_input("HF Token", value="", type="password", placeholder="hf_... (paste here or set HF_TOKEN)", help="Get token at hf.co/settings/tokens")
        hf_token = hf_token or (token_input.strip() if token_input else "")
        model_id = st.selectbox("Model", HF_MODELS, index=0)
        st.markdown("**Supported languages**")
        st.caption("Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, Ruby, PHP, Swift, Kotlin, Scala, R, SQL")
        st.markdown("---")
        st.markdown("**Analysis parameters**")
        st.caption("• Functions & Methods  • Endpoints / APIs  • Dependencies  • Optimizations  • Potential Issues")
        st.markdown("---")
        st.caption("Code Analyzer v1.0")
    
    st.markdown("""
    <div class="hero">
        <h1>Code Analyzer</h1>
        <p>Clone any Git repository and analyze code across 5 dimensions: Functions & Methods, Endpoints/APIs, Dependencies, Optimizations, and Potential Issues. Enterprise-grade structured reports.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "repo_path" not in st.session_state:
        st.session_state.repo_path = None
    
    with st.container():
        st.markdown('<p class="section-title">Step 1 — Clone Repository</p>', unsafe_allow_html=True)
        st.caption("Paste a GitHub, GitLab, or Bitbucket URL. Optionally specify a subfolder for faster checkout.")
        repo_url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo.git", label_visibility="visible", key="repo_url")
        sparse_path = st.text_input("Subfolder (optional)", placeholder="e.g. src, packages/api — leave empty for full clone", key="sparse", label_visibility="visible")
        clone_clicked = st.button("Clone repository", type="primary")
    
    if clone_clicked and repo_url:
        with st.spinner("Cloning..."):
            temp_dir = tempfile.mkdtemp(prefix="code_analyzer_")
            success, msg = clone_repo(repo_url, temp_dir, sparse_path or None)
            if success:
                st.session_state.repo_path = temp_dir
                st.success(msg)
            else:
                st.error(msg)
    
    if st.session_state.repo_path and os.path.exists(st.session_state.repo_path):
        repo_path = st.session_state.repo_path
        folders = get_folder_structure(repo_path)
        st.success(f"Repository loaded — {len(folders)} folders ready for analysis")
        
        with st.container():
            st.markdown('<p class="section-title">Step 2 — Analyze</p>', unsafe_allow_html=True)
            st.caption("All code will be analyzed on 5 parameters. Click Run Analysis to start.")
        
        code_files = []
        seen = set()
        for fpath, content in get_code_files(repo_path):
            if fpath not in seen:
                seen.add(fpath)
                code_files.append((fpath, content))
        if not code_files:
            st.warning("No code files found.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files", len(code_files))
            with col2:
                st.metric("Lines", f"{sum(c.count(chr(10)) + 1 for _, c in code_files):,}")
            with col3:
                st.metric("Functions", sum(len(extract_functions_from_code(f, c)) for f, c in code_files))
            
            with st.expander("View files in scope", expanded=False):
                for fpath, content in code_files[:20]:
                    items = extract_functions_from_code(fpath, content)
                    if items:
                        fn_list = ", ".join(f"`{i['name']}`" for i in items[:8])
                        if len(items) > 8:
                            fn_list += f" +{len(items)-8} more"
                        st.markdown(f"**{fpath}** — {fn_list}")
                    else:
                        st.markdown(f"**{fpath}**")
                if len(code_files) > 20:
                    st.caption(f"+ {len(code_files)-20} more files")
            
            run_btn = st.button("Run Analysis", type="primary", disabled=not hf_token)
            if run_btn:
                if not hf_token:
                    st.error("Add HF_TOKEN: paste in sidebar or set HF_TOKEN env var.")
                    st.stop()
                start = time.time()
                hf_token = hf_token or get_hf_token()
                batches = split_into_batches(code_files, n_batches=4)
                progress = st.progress(0, text="Analyzing in parallel...")
                results = []
                total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                error_msg = None
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(_analyze_batch, b, model_id, hf_token): i for i, b in enumerate(batches)}
                    for i, future in enumerate(as_completed(futures)):
                        batch_idx = futures[future]
                        try:
                            data, usage, err = future.result()
                            if data:
                                results.append(data)
                            if err and not error_msg:
                                error_msg = err
                            for k in total_usage:
                                total_usage[k] += usage.get(k, 0)
                        except Exception as e:
                            error_msg = str(e)
                        progress.progress((i + 1) / len(batches), text=f"Batch {batch_idx + 1}/{len(batches)} done")
                progress.empty()
                elapsed = time.time() - start
                data = merge_analysis_results(results) if results else None
                success = data is not None and any(data.get(k) for k in ["functions", "endpoints", "dependencies", "optimizations", "issues"])
                
                if data and (data.get("functions") or data.get("endpoints") or data.get("dependencies") or data.get("optimizations") or data.get("issues")):
                    param_keys = [p[0] for p in ANALYSIS_PARAMS]
                    meta = {"model": model_id, "time_seconds": round(elapsed, 1), "usage": total_usage, "scope": "all", "batches": len(batches)}
                    full_json = {"metadata": meta, "analysis": {k: data.get(k, []) for k in param_keys}}
                    json_str = json.dumps(full_json, indent=2)
                    
                    st.markdown("---")
                    st.markdown("### Analysis Report")
                    
                    # Summary metrics
                    st.markdown("#### Summary")
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    with sm1:
                        st.metric("Model", model_id.split("/")[-1])
                    with sm2:
                        st.metric("Duration", f"{elapsed:.1f}s")
                    with sm3:
                        st.metric("Prompt tokens", total_usage.get("prompt_tokens", "-"))
                    with sm4:
                        st.metric("Completion tokens", total_usage.get("completion_tokens", "-"))
                    
                    st.markdown("---")
                    
                    # Tabs for each analysis section
                    tab_names = [f"{p[2]} {p[1]} ({len(data.get(p[0], []))})" for p in ANALYSIS_PARAMS]
                    result_tabs = st.tabs(tab_names)
                    for tab, (param_key, title, icon) in zip(result_tabs, ANALYSIS_PARAMS):
                        with tab:
                            items = data.get(param_key, [])
                            if not items:
                                st.info("No items found in this category.")
                            else:
                                for i, item in enumerate(items):
                                    with st.container():
                                        if isinstance(item, dict):
                                            header = item.get("name") or item.get("path") or item.get("area") or item.get("type") or f"Item {i + 1}"
                                            st.markdown(f"**{header}**")
                                            skip = {"name", "path", "area", "type"}
                                            details = []
                                            for k, v in item.items():
                                                if k in skip or v is None or v == "":
                                                    continue
                                                if isinstance(v, list):
                                                    details.append(f"**{k}:** {', '.join(str(x) for x in v)}")
                                                else:
                                                    details.append(f"**{k}:** {v}")
                                            if details:
                                                st.markdown("  \n\n".join(f"- {d}" for d in details))
                                        else:
                                            st.markdown(f"- {item}")
                                        if i < len(items) - 1:
                                            st.divider()
                    
                    st.markdown("---")
                    st.markdown("#### Export")
                    with st.expander("View / Download JSON", expanded=False):
                        st.caption("Structured output for integration or further processing.")
                        st.code(json_str, language="json")
                        st.download_button("Download JSON", json_str, file_name="analysis_report.json", mime="application/json")
                else:
                    if error_msg:
                        st.error(error_msg)
                    elif not results:
                        st.error("Analysis failed. No results returned.")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Features**")
        st.caption("Git clone (full or sparse) · Multi-language support · Structured JSON export · Downloadable reports")
    with col2:
        st.markdown("**Workflow**")
        st.caption("Clone repository → Run analysis → View results by section → Export JSON")
    with col3:
        st.markdown("**Stack**")
        st.caption("Streamlit · Hugging Face · GitPython")


if __name__ == "__main__":
    main()
