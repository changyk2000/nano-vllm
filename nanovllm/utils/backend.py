import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


@dataclass
class Clause:
    kind: str
    value: str
    operator: str | None = None
    expected: str | None = None


class BackendAPI:
    CLAUSE_SPLIT_PATTERN = re.compile(r"\s+(and|or|&&|\|\|)\s+", re.IGNORECASE)
    LLM_PATTERN = re.compile(
        r"LLM\((?P<quote>['\"])(?P<prompt>.*?)\1\)\s*(?P<op>==|!=)\s*(?P<label_quote>['\"])(?P<label>.*?)\4",
        re.IGNORECASE,
    )
    TEMPLATE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

    def __init__(self) -> None:
        print("init backend")
        path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
        self.llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
        self.base_sampling = SamplingParams(temperature=1, max_tokens=1)
        self.data: pd.DataFrame | None = None
        self.data_path: str | None = None
        self.text_field: str | None = None
        self.index_ready = False
        self.current_sparsity: float | None = None
        self.index_limit: int | None = None
        self.analytics: dict[str, list[dict[str, Any]]] = {"indexes": [], "queries": []}

    # ------------------------------------------------------------------
    # Data Loading & Indexing
    # ------------------------------------------------------------------
    def load_data(self, data_path: str) -> dict[str, Any]:
        df = pd.read_csv(data_path)
        df["__id"] = range(len(df))
        self.data = df
        self.data_path = data_path
        self.index_ready = False
        self.text_field = self._infer_text_field(df)
        self.analytics["queries"].clear()
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "text_field": self.text_field,
        }

    def _infer_text_field(self, df: pd.DataFrame) -> str:
        for column in df.columns:
            if column == "__id":
                continue
            if pd.api.types.is_string_dtype(df[column]):
                return column
        return df.columns[0]

    def build_index(
        self,
        sparsity: float,
        field: str | None = None,
        limit: int | None = 1000,
    ) -> dict[str, Any]:
        self._ensure_data_loaded()
        field = field or self.text_field
        if field is None or field not in self.data.columns:  # type: ignore[attr-defined]
            raise ValueError("Invalid text field for index building.")

        # Reset KV cache index
        self.llm.kv_cache_index.kv_cache_index = {}
        subset = self.data[["__id", field]]  # type: ignore[index]
        if limit is not None:
            subset = subset.head(limit)

        samples: list[tuple[int, str]] = []
        for _, row in subset.iterrows():
            text_id = int(row["__id"])
            text = str(row[field])
            prompt = f"{text}\n "
            samples.append((text_id, prompt))

        sp = SamplingParams(
            temperature=self.base_sampling.temperature,
            max_tokens=self.base_sampling.max_tokens,
        )
        sp.task_str_len = 1

        self.llm.generate(
            samples,
            sp,
            use_index=True,
            use_tqdm=False,
            pruning=True,
            sparsity=sparsity,
        )

        index_size = self._estimate_index_size()
        self.index_ready = True
        self.text_field = field
        self.current_sparsity = sparsity
        self.index_limit = limit
        meta = {
            "field": field,
            "rows_indexed": len(samples),
            "sparsity": sparsity,
            "index_size_bytes": index_size,
        }
        self._append_index_meta(meta)
        return meta

    def _estimate_index_size(self) -> int:
        total = 0
        for item in self.llm.kv_cache_index.kv_cache_index.values():
            kv_tensor = item.get("kv") if isinstance(item, dict) else item
            if hasattr(kv_tensor, "nelement") and hasattr(kv_tensor, "element_size"):
                total += int(kv_tensor.nelement()) * int(kv_tensor.element_size())  # type: ignore[attr-defined]
        return total

    def _append_index_meta(self, meta: dict[str, Any]) -> None:
        self.analytics["indexes"].append({
            "timestamp": time.time(),
            **meta,
        })
        self.analytics["indexes"] = self.analytics["indexes"][-20:]

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------
    def query(self, query: str, use_index: bool, limit: int | None = 50) -> dict[str, Any]:
        self._ensure_data_loaded()
        df = self.data.copy()  # type: ignore[assignment]
        clauses, connectors = self._parse_query(query)

        clause_masks: list[pd.Series] = []
        llm_metrics: list[dict[str, Any]] = []
        expr_metrics: list[dict[str, Any]] = []

        if not clauses:
            clause_masks.append(pd.Series([True] * len(df), index=df.index))

        for clause in clauses:
            if clause.kind == "llm":
                mask, metrics = self._execute_llm_clause(df, clause, use_index)
                clause_masks.append(mask)
                llm_metrics.append(metrics)
            else:
                mask, metrics = self._execute_expr_clause(df, clause.value)
                clause_masks.append(mask)
                expr_metrics.append(metrics)

        final_mask = clause_masks[0]
        for connector, mask in zip(connectors, clause_masks[1:]):
            if connector in {"and", "&&"}:
                final_mask = final_mask & mask
            else:
                final_mask = final_mask | mask

        final_mask = final_mask.fillna(False)
        filtered = df.loc[final_mask]
        total_matches = len(filtered)
        preview = filtered if limit is None else filtered.head(limit)

        inference_time = sum(m.get("latency_s", 0.0) for m in llm_metrics) + sum(
            m.get("latency_s", 0.0) for m in expr_metrics
        )

        trace = self._build_trace_events(llm_metrics, expr_metrics)
        preview_for_user = preview.copy()
        if "__id" in preview_for_user.columns:
            preview_for_user = preview_for_user.rename(columns={"__id": "row_id"})
        results = preview_for_user.to_dict(orient="records")
        metadata = {
            "total_rows": len(df),
            "matched_rows": total_matches,
            "returned_rows": len(preview),
            "use_index": any(m.get("use_index") for m in llm_metrics),
            "text_field": self.text_field,
            "limit": limit,
        }

        signature = hashlib.sha1(query.encode("utf-8")).hexdigest()
        self._append_query_log(
            query,
            signature,
            llm_metrics,
            filtered["__id"].tolist() if "__id" in filtered.columns else [],
        )

        return {
            "results": results,
            "metadata": metadata,
            "inference_time": f"{inference_time:.4f} seconds",
            "trace_data": trace,
        }

    def _execute_llm_clause(
        self, df: pd.DataFrame, clause: Clause, use_index: bool
    ) -> tuple[pd.Series, dict[str, Any]]:
        start = time.perf_counter()
        if df.empty:
            return pd.Series([], dtype=bool), {
                "latency_s": 0.0,
                "transfer_ms": 0.0,
                "compute_ms": 0.0,
                "use_index": False,
                "prompt": clause.value,
            }

        effective_index = bool(use_index and self.index_ready)
        tuple_prompts: list[tuple[int, str]] = []
        string_prompts: list[str] = []
        params: list[SamplingParams] = []
        order: list[Any] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            context_value = str(row_dict.get(self.text_field, "")) if self.text_field else ""
            rendered_prompt = self._render_prompt_template(clause.value, row_dict)
            # Guarantee at least one character for the task tail
            task_suffix = rendered_prompt or "Answer:"
            full_prompt = f"{context_value}\n\n{task_suffix}"
            if effective_index:
                tuple_prompts.append((int(row_dict.get("__id", idx)), full_prompt))
            else:
                string_prompts.append(full_prompt)
            sp = SamplingParams(
                temperature=self.base_sampling.temperature,
                max_tokens=self.base_sampling.max_tokens,
            )
            sp.task_str_len = max(len(task_suffix), 1)
            params.append(sp)
            order.append(idx)

        prompt_payload: list[str] | list[tuple[int, str]] = (
            tuple_prompts if effective_index else string_prompts
        )
        outputs = self.llm.generate(
            prompt_payload,
            params,
            use_index=effective_index,
            use_tqdm=False,
            pruning=False,
            sparsity=self.current_sparsity or 0.9,
        )

        latency = time.perf_counter() - start
        stats = getattr(self.llm, "last_run_stats", None) or {}
        transfer_ms = stats.get("avg_transfer_ms", 0.0)
        compute_ms = stats.get("avg_compute_ms", latency * 1000)

        normalized_expected = self._normalize_prediction(clause.expected or "")
        series_data = []
        for idx, output in zip(order, outputs):
            prediction = self._normalize_prediction(output.get("text", ""))
            if clause.operator == "!=":
                series_data.append(prediction != normalized_expected)
            else:
                series_data.append(prediction == normalized_expected)

        mask = pd.Series(series_data, index=order)
        metrics = {
            "latency_s": latency,
            "transfer_ms": transfer_ms,
            "compute_ms": compute_ms,
            "use_index": effective_index,
            "prompt": clause.value,
        }
        return mask, metrics

    def _execute_expr_clause(self, df: pd.DataFrame, expr: str) -> tuple[pd.Series, dict[str, Any]]:
        start = time.perf_counter()
        result = df.eval(expr, engine="python")
        duration = time.perf_counter() - start
        if isinstance(result, pd.Series):
            mask = result.reindex(df.index).fillna(False).astype(bool)
        else:
            mask = pd.Series(bool(result), index=df.index)
        metrics = {
            "latency_s": duration,
            "label": expr,
        }
        return mask, metrics

    def _parse_query(self, query: str) -> tuple[list[Clause], list[str]]:
        parts = self.CLAUSE_SPLIT_PATTERN.split(query)
        clauses: list[Clause] = []
        connectors: list[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                clause = part.strip().strip("()")
                if not clause:
                    continue
                match = self.LLM_PATTERN.match(clause)
                if match:
                    clauses.append(
                        Clause(
                            kind="llm",
                            value=match.group("prompt"),
                            operator=match.group("op"),
                            expected=match.group("label"),
                        )
                    )
                else:
                    clauses.append(Clause(kind="expr", value=clause))
            else:
                connectors.append(part.strip().lower())
        return clauses, connectors

    def _render_prompt_template(self, template: str, row: dict[str, Any]) -> str:
        row_with_defaults = {**row}
        if self.text_field and self.text_field not in row_with_defaults:
            row_with_defaults[self.text_field] = ""
        if self.text_field:
            default_text = str(row_with_defaults.get(self.text_field, ""))
        else:
            default_text = ""
        row_with_defaults.setdefault("text", default_text)

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(row_with_defaults.get(key, ""))

        return self.TEMPLATE_PATTERN.sub(repl, template)

    def _normalize_prediction(self, text: str) -> str:
        return text.strip().strip("\n").lower()

    def _build_trace_events(
        self,
        llm_metrics: Iterable[dict[str, Any]],
        expr_metrics: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        events = []
        ts = 0
        pid = 1
        tid_transfer = 1
        tid_compute = 2
        tid_filter = 3

        for metric in llm_metrics:
            transfer = max(metric.get("transfer_ms", 0.0), 0.0) * 1000
            compute = max(metric.get("compute_ms", 0.0), 0.0) * 1000
            if transfer:
                events.append(
                    {
                        "ph": "X",
                        "name": "KV Cache Transfer",
                        "ts": ts,
                        "dur": transfer,
                        "pid": pid,
                        "tid": tid_transfer,
                        "args": {"prompt": metric.get("prompt")},
                    }
                )
                ts += transfer
            if compute:
                events.append(
                    {
                        "ph": "X",
                        "name": "Computation",
                        "ts": ts,
                        "dur": compute,
                        "pid": pid,
                        "tid": tid_compute,
                        "args": {"prompt": metric.get("prompt")},
                    }
                )
                ts += compute

        for metric in expr_metrics:
            dur = metric.get("latency_s", 0.0) * 1_000_000
            if dur == 0:
                continue
            events.append(
                {
                    "ph": "X",
                    "name": "Pandas Filter",
                    "ts": ts,
                    "dur": dur,
                    "pid": pid,
                    "tid": tid_filter,
                    "args": {"expr": metric.get("label")},
                }
            )
            ts += dur

        return {"traceEvents": events}

    def _append_query_log(
        self,
        query: str,
        signature: str,
        llm_metrics: list[dict[str, Any]],
        result_ids: list[int],
    ) -> None:
        latency = sum(metric.get("latency_s", 0.0) for metric in llm_metrics)
        entry = {
            "timestamp": time.time(),
            "query": query,
            "signature": signature,
            "use_index": any(m.get("use_index") for m in llm_metrics),
            "latency_ms": latency * 1000,
            "transfer_ms": sum(m.get("transfer_ms", 0.0) for m in llm_metrics),
            "compute_ms": sum(m.get("compute_ms", 0.0) for m in llm_metrics),
            "result_ids": result_ids,
            "sparsity": self.current_sparsity,
        }
        self.analytics["queries"].append(entry)
        self.analytics["queries"] = self.analytics["queries"][-50:]

    # ------------------------------------------------------------------
    # Analytics & Plotting
    # ------------------------------------------------------------------
    def _get_pics_dir(self) -> str:
        """Get the pics directory path, creating it if it doesn't exist."""
        # Use ui/static/pics for Flask to serve the images
        pics_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'ui', 'static', 'pics'
        )
        os.makedirs(pics_dir, exist_ok=True)
        return pics_dir

    def _save_latency_bar_chart(self, latency_bar: dict[str, float], pics_dir: str) -> str | None:
        """Generate and save latency comparison bar chart."""
        if not latency_bar.get('with_index') and not latency_bar.get('without_index'):
            return None
        
        try:
            plt.figure(figsize=(8, 6))
            categories = ['With Index', 'Without Index']
            values = [latency_bar.get('with_index', 0), latency_bar.get('without_index', 0)]
            
            colors = ['#38bdf8', '#f97316']
            bars = plt.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f} ms',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
            plt.title('Latency Comparison', fontsize=14, fontweight='bold', pad=20)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            filepath = os.path.join(pics_dir, 'latency_comparison.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            return 'latency_comparison.png'
        except (IOError, OSError) as e:
            # Handle file I/O errors (e.g., insufficient disk space, permissions)
            print(f"Error saving latency chart: {e}")
            plt.close()
            return None

    def _save_diff_scatter_chart(self, diff_points: list[dict[str, Any]], pics_dir: str) -> str | None:
        """Generate and save result drift scatter plot."""
        if not diff_points:
            return None
        
        try:
            plt.figure(figsize=(8, 6))
            x_values = [p.get('diff_ratio', 0) for p in diff_points]
            y_values = [p.get('sparsity', 0) for p in diff_points]
            
            scatter = plt.scatter(x_values, y_values, c='#f97316', s=100, alpha=0.7, 
                                edgecolors='#ea580c', linewidth=2)
            
            plt.xlabel('Result Difference Ratio', fontsize=12, fontweight='bold')
            plt.ylabel('Sparsity', fontsize=12, fontweight='bold')
            plt.title('Result Drift Analysis', fontsize=14, fontweight='bold', pad=20)
            plt.grid(alpha=0.3, linestyle='--')
            
            # Format axes as percentages
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y*100:.0f}%'))
            
            plt.tight_layout()
            filepath = os.path.join(pics_dir, 'result_drift.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            return 'result_drift.png'
        except (IOError, OSError) as e:
            print(f"Error saving diff scatter chart: {e}")
            plt.close()
            return None

    def _save_sparsity_timeline_chart(self, sparsity_curve: list[dict[str, Any]], pics_dir: str) -> str | None:
        """Generate and save sparsity timeline chart."""
        if not sparsity_curve:
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            # Filter out entries with missing or invalid timestamps
            valid_entries = [p for p in sparsity_curve if p.get('timestamp', 0) > 0]
            if not valid_entries:
                return None
                
            timestamps = [datetime.fromtimestamp(p['timestamp']) for p in valid_entries]
            sparsity_values = [p.get('sparsity', 0) for p in valid_entries]
            
            plt.plot(timestamps, sparsity_values, marker='o', color='#10b981', 
                    linewidth=2.5, markersize=8, markerfacecolor='#047857', 
                    markeredgecolor='white', markeredgewidth=2)
            
            plt.xlabel('Index Build Time', fontsize=12, fontweight='bold')
            plt.ylabel('Sparsity', fontsize=12, fontweight='bold')
            plt.title('Sparsity Timeline', fontsize=14, fontweight='bold', pad=20)
            plt.grid(alpha=0.3, linestyle='--')
            
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y*100:.0f}%'))
            
            # Format x-axis with better date formatting
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            filepath = os.path.join(pics_dir, 'sparsity_timeline.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            return 'sparsity_timeline.png'
        except (IOError, OSError, ValueError) as e:
            print(f"Error saving sparsity timeline chart: {e}")
            plt.close()
            return None

    def _save_kv_timeline_chart(self, kv_timeline: list[dict[str, Any]], pics_dir: str) -> str | None:
        """Generate and save KV latency timeline chart."""
        if not kv_timeline:
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            # Filter out entries with missing or invalid timestamps
            valid_entries = [p for p in kv_timeline if p.get('timestamp', 0) > 0]
            if not valid_entries:
                return None
                
            timestamps = [datetime.fromtimestamp(p['timestamp']) for p in valid_entries]
            transfer_values = [p.get('transfer_ms', 0) for p in valid_entries]
            compute_values = [p.get('compute_ms', 0) for p in valid_entries]
            
            plt.plot(timestamps, transfer_values, marker='o', color='#2563eb', 
                    linewidth=2, markersize=6, label='Transfer Time', 
                    markerfacecolor='#2563eb', markeredgecolor='white', markeredgewidth=1.5)
            plt.plot(timestamps, compute_values, marker='s', color='#facc15', 
                    linewidth=2, markersize=6, label='Compute Time',
                    markerfacecolor='#facc15', markeredgecolor='white', markeredgewidth=1.5)
            
            plt.xlabel('Query Time', fontsize=12, fontweight='bold')
            plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
            plt.title('KV Cache Latency Timeline', fontsize=14, fontweight='bold', pad=20)
            plt.legend(loc='best', frameon=True, shadow=True)
            plt.grid(alpha=0.3, linestyle='--')
            
            # Format x-axis with better date formatting
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            filepath = os.path.join(pics_dir, 'kv_latency_timeline.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            return 'kv_latency_timeline.png'
        except (IOError, OSError, ValueError) as e:
            print(f"Error saving KV timeline chart: {e}")
            plt.close()
            return None

    def analyse(self, _data: Any | None = None) -> dict[str, Any]:
        queries = self.analytics["queries"]
        indexes = self.analytics["indexes"]
        latency_with = [q["latency_ms"] for q in queries if q.get("use_index")]
        latency_without = [q["latency_ms"] for q in queries if not q.get("use_index")]

        latency_bar = {
            "with_index": sum(latency_with) / len(latency_with) if latency_with else 0,
            "without_index": sum(latency_without) / len(latency_without)
            if latency_without
            else 0,
        }

        diff_points = []
        grouped: dict[str, dict[bool, dict[str, Any]]] = {}
        for entry in queries:
            grouped.setdefault(entry["signature"], {})[bool(entry["use_index"])] = entry
        for group in grouped.values():
            if True in group and False in group:
                idx_set = set(group[True]["result_ids"])
                base_set = set(group[False]["result_ids"])
                union = idx_set | base_set
                if not union:
                    continue
                diff = 1 - (len(idx_set & base_set) / len(union))
                diff_points.append(
                    {
                        "diff_ratio": diff,
                        "sparsity": group[True].get("sparsity"),
                        "query": group[True]["query"][:80],
                    }
                )

        sparsity_curve = [
            {
                "timestamp": item["timestamp"],
                "sparsity": item.get("sparsity"),
                "index_size_kb": (item.get("index_size_bytes", 0) / 1024),
            }
            for item in indexes
        ]

        kv_timeline = [
            {
                "timestamp": q["timestamp"],
                "transfer_ms": q.get("transfer_ms"),
                "compute_ms": q.get("compute_ms"),
                "use_index": q.get("use_index"),
            }
            for q in queries
            if q.get("transfer_ms") or q.get("compute_ms")
        ]

        # Generate plots and save to pics directory
        pics_dir = self._get_pics_dir()
        plot_files = {
            "latency_bar": self._save_latency_bar_chart(latency_bar, pics_dir),
            "diff_scatter": self._save_diff_scatter_chart(diff_points, pics_dir),
            "sparsity_timeline": self._save_sparsity_timeline_chart(sparsity_curve, pics_dir),
            "kv_timeline": self._save_kv_timeline_chart(kv_timeline, pics_dir),
        }

        return {
            "latency_bar": latency_bar,
            "diff_points": diff_points,
            "sparsity_curve": sparsity_curve,
            "kv_timeline": kv_timeline,
            "plot_files": plot_files,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_data_loaded(self) -> None:
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Please upload a CSV first.")

