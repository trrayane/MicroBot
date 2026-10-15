"""Bilan complet des ressources utilisées pendant l'ingestion d'un document.

Mesure : CPU (par cœur et global), RAM (avant / pic / après), temps par phase
(parse, chunk, embed, index) et taille des structures en mémoire.

Usage :
    python benchmarks/ingestion_profile.py                    # petit fichier texte
    python benchmarks/ingestion_profile.py --file x.xlsx      # ton propre fichier
    python benchmarks/ingestion_profile.py --size medium      # taille prédéfinie
    python benchmarks/ingestion_profile.py --iterations 5     # moyenne sur N essais
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mc_rag.infrastructure.embeddings.local_embedder import get_embedder  # noqa: E402
from mc_rag.infrastructure.lexical_store.bm25_index import BM25Index  # noqa: E402
from mc_rag.infrastructure.vector_store.chroma_store import ChromaStore  # noqa: E402
from mc_rag.rag.chunking import chunk_document  # noqa: E402

# --------------------------------------------------------------------------- #
# Sampling thread
# --------------------------------------------------------------------------- #


@dataclass
class ResourceSample:
    """One point-in-time snapshot of CPU and memory."""

    cpu_percent: float
    cpu_per_core: list[float]
    rss_mb: float
    vms_mb: float
    timestamp: float


@dataclass
class ResourceMonitor:
    """Background thread sampling CPU/RAM every `interval` seconds."""

    process: psutil.Process
    interval: float = 0.05
    samples: list[ResourceSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cpu = self.process.cpu_percent(interval=None)
                mem = self.process.memory_info()
                self.samples.append(
                    ResourceSample(
                        cpu_percent=cpu,
                        cpu_per_core=[cpu],  # per-process, not per-core
                        rss_mb=mem.rss / 1024 / 1024,
                        vms_mb=mem.vms / 1024 / 1024,
                        timestamp=time.time(),
                    )
                )
            except psutil.NoSuchProcess:
                break
            self._stop.wait(self.interval)

    @property
    def peak_rss_mb(self) -> float:
        return max((s.rss_mb for s in self.samples), default=0.0)

    @property
    def min_rss_mb(self) -> float:
        return min((s.rss_mb for s in self.samples), default=0.0)

    @property
    def peak_cpu_percent(self) -> float:
        return max((s.cpu_percent for s in self.samples), default=0.0)

    @property
    def avg_cpu_percent(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.mean(s.cpu_percent for s in self.samples)

    @property
    def peak_core_percent(self) -> float:
        """Highest usage seen on any single core."""
        return max(
            (c for s in self.samples for c in s.cpu_per_core),
            default=0.0,
        )

    @property
    def sample_count(self) -> int:
        return len(self.samples)


# --------------------------------------------------------------------------- #
# Phase timing
# --------------------------------------------------------------------------- #


@dataclass
class PhaseResult:
    """Duration and resources for one ingestion phase."""

    name: str
    duration_s: float
    peak_rss_mb: float
    result_size: int = 0


@contextmanager
def measure_phase(
    name: str,
    monitor: ResourceMonitor,
    results: list[PhaseResult],
    size_fn: Callable[[Any], int] | None = None,
) -> Any:
    """Context manager that times a phase and records peak RSS."""
    monitor.samples.clear()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        peak = max(mem_before, mem_after, monitor.peak_rss_mb)
        results.append(
            PhaseResult(name=name, duration_s=elapsed, peak_rss_mb=peak)
        )


# --------------------------------------------------------------------------- #
# Test documents
# --------------------------------------------------------------------------- #


def make_text_doc(size: str) -> str:
    """Generate a synthetic text document of the given size."""
    base = (
        "Le Micro Club est un club informatique a l'Universite Mohamed Khider "
        "de Biskra. Il organise des formations, des hackathons et des ateliers "
        "techniques pour les etudiants. Les membres apprennent le developpement "
        "web, la cyberscurite et l'intelligence artificielle. "
    )
    multiplier = {"tiny": 1, "small": 10, "medium": 50, "large": 200}.get(size, 10)
    return base * multiplier


def make_xlsx_doc() -> Path | None:
    """Return the real student XLSX if present, else None."""
    candidates = [
        Path("data/uploads/public/M1_IV_A_Liste_Affichage_.xlsx"),
        Path("data/uploads/public") / "M1_IV_A_Liste_Affichage_.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# Benchmark runner
# --------------------------------------------------------------------------- #


@dataclass
class IngestionBenchmark:
    """Full ingestion benchmark for one document."""

    label: str
    file_size_kb: float
    phases: list[PhaseResult]
    total_rss_before_mb: float
    total_rss_after_mb: float
    total_duration_s: float

    def print(self) -> None:
        w = 62
        print()
        print("=" * w)
        print(f"  {self.label}")
        print("=" * w)
        print(f"  Taille fichier       : {self.file_size_kb:.1f} KB")
        print(f"  Duree totale         : {self.total_duration_s:.3f} s")
        print(f"  RAM avant            : {self.total_rss_before_mb:.1f} MB")
        print(f"  RAM apres            : {self.total_rss_after_mb:.1f} MB")
        print(f"  RAM nette (+/-)      : "
              f"{self.total_rss_after_mb - self.total_rss_before_mb:+.1f} MB")
        print("-" * w)
        print(f"  {'Phase':<20} {'Temps':>10} {'Pic RSS':>10} {'Taille':>12}")
        print("-" * w)
        for p in self.phases:
            size_str = f"{p.result_size} items" if p.result_size else "-"
            print(f"  {p.name:<20} {p.duration_s:>9.3f}s {p.peak_rss_mb:>9.1f}M {size_str:>12}")
        print("=" * w)


def run_benchmark(
    label: str,
    *,
    xlsx_path: Path | None = None,
    text_content: str | None = None,
    temp_dir: Path | None = None,
) -> IngestionBenchmark:
    """Run a full ingestion benchmark and return the results."""
    process = psutil.Process(os.getpid())

    # Prepare the file
    if xlsx_path is not None:
        data = xlsx_path.read_bytes()
        file_size_kb = len(data) / 1024
        suffix = ".xlsx"
        source_path = xlsx_path
    else:
        content = text_content or make_text_doc("small")
        data = content.encode("utf-8")
        file_size_kb = len(data) / 1024
        suffix = ".txt"
        tmp = (temp_dir or Path(tempfile.gettempdir())) / "bench_ingest.txt"
        tmp.write_text(content, encoding="utf-8")
        source_path = tmp

    monitor = ResourceMonitor(process)
    phases: list[PhaseResult] = []

    rss_before = process.memory_info().rss / 1024 / 1024
    t_start = time.perf_counter()

    # --- Phase 1: parse ---
    monitor.start()
    with measure_phase("1. Parse", monitor, phases):
        if suffix == ".xlsx":
            from mc_rag.infrastructure.parsers import parse_xlsx as _parse
            document = _parse(source_path)
        else:
            from mc_rag.infrastructure.parsers import ParsedDocument
            document = ParsedDocument(
                text=source_path.read_text(encoding="utf-8"),
                source=str(source_path),
                format="text",
                title="bench",
            )
    monitor.stop()

    # --- Phase 2: chunk ---
    monitor.start()
    with measure_phase("2. Chunk", monitor, phases):
        chunks = chunk_document(document)
    monitor.stop()
    if phases:
        phases[-1].result_size = len(chunks)

    # --- Phase 3a: load embedding model ---
    monitor.start()
    with measure_phase("3a. Model load", monitor, phases):
        embedder = get_embedder()
        _ = embedder.model  # force lazy load
    monitor.stop()

    # --- Phase 3b: embed texts ---
    monitor.start()
    with measure_phase("3b. Embed texts", monitor, phases):
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)
    monitor.stop()
    if phases:
        phases[-1].result_size = len(embeddings)

    # --- Phase 4: write to Chroma ---
    monitor.start()
    with measure_phase("4. Chroma write", monitor, phases):
        chroma = ChromaStore()
        chroma.add_chunks(chunks, embeddings)
        chroma.delete_document("bench-doc")
    monitor.stop()
    if phases:
        phases[-1].result_size = len(chunks)

    # --- Phase 5: write to BM25 ---
    monitor.start()
    with measure_phase("5. BM25 write", monitor, phases):
        from mc_rag.infrastructure.lexical_store.bm25_index import IndexEntry
        bm25 = BM25Index()
        entries = [
            IndexEntry(chunk_id=c.chunk_id, text=c.text, metadata=c.metadata)
            for c in chunks
        ]
        bm25.upsert(entries)
        bm25.delete_document("bench-doc")
    monitor.stop()

    total_duration = time.perf_counter() - t_start
    rss_after = process.memory_info().rss / 1024 / 1024

    return IngestionBenchmark(
        label=label,
        file_size_kb=file_size_kb,
        phases=phases,
        total_rss_before_mb=rss_before,
        total_rss_after_mb=rss_after,
        total_duration_s=total_duration,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Bilan des ressources d'ingestion")
    parser.add_argument("--file", type=Path, help="Fichier XLSX a tester")
    parser.add_argument(
        "--size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Taille du document texte synthetique",
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Nombre d'essais (moyenne si > 1)",
    )
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} | psutil {psutil.__version__}")
    print(f"CPU : {psutil.cpu_count()} coeurs | "
          f"RAM totale : {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Process PID : {os.getpid()}")

    results: list[IngestionBenchmark] = []

    for i in range(args.iterations):
        if args.file:
            label = f"XLSX reel : {args.file.name} (essai {i+1}/{args.iterations})"
            results.append(run_benchmark(label, xlsx_path=args.file))
        else:
            content = make_text_doc(args.size)
            label = f"Texte {args.size} : {len(content)} caracteres (essai {i+1}/{args.iterations})"
            results.append(run_benchmark(label, text_content=content))

    # Print each run
    for r in results:
        r.print()

    # Aggregate if multiple iterations
    if len(results) > 1:
        print("\n" + "=" * 62)
        print(f"  MOYENNE sur {len(results)} essais")
        print("=" * 62)
        avg_dur = statistics.mean(r.total_duration_s for r in results)
        avg_rss = statistics.mean(
            max(p.peak_rss_mb for p in r.phases) for r in results
        )
        avg_delta = statistics.mean(
            r.total_rss_after_mb - r.total_rss_before_mb for r in results
        )
        print(f"  Duree moyenne  : {avg_dur:.3f} s")
        print(f"  Pic RSS moyen  : {avg_rss:.1f} MB")
        print(f"  Delta RAM moyen: {avg_delta:+.1f} MB")
        print("=" * 62)

    # Warm vs cold comparison when running multiple iterations
    if len(results) >= 2:
        cold = results[0]
        warm = results[1]
        print("\n" + "=" * 62)
        print("  FROID vs CHAUD (meme processus)")
        print("=" * 62)
        print(f"  {'':20} {'Froid (1er)':>12} {'Chaud (2e)':>12} {'Gain':>10}")
        print("-" * 62)
        for i in range(min(len(cold.phases), len(warm.phases))):
            pc = cold.phases[i]
            pw = warm.phases[i]
            if pc.duration_s > 0.01:
                gain = (1 - pw.duration_s / pc.duration_s) * 100
                print(
                    f"  {pc.name:<20} {pc.duration_s:>11.3f}s "
                    f"{pw.duration_s:>11.3f}s {gain:>9.0f}%"
                )
        print("-" * 62)
        print(
            f"  {'TOTAL':<20} {cold.total_duration_s:>11.3f}s "
            f"{warm.total_duration_s:>11.3f}s "
            f"{(1 - warm.total_duration_s / cold.total_duration_s) * 100:>9.0f}%"
        )
        print("=" * 62)


if __name__ == "__main__":
    main()
