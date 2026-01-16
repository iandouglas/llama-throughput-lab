import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tests.llama_server_test_utils import (
    extract_token_count,
    post_json,
    start_llama_servers,
    start_nginx_round_robin,
)


def _parse_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    return [int(item) for item in parts if item]


def _parse_optional_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    result = []
    for item in parts:
        if not item:
            continue
        if item.lower() == "default":
            result.append(None)
        else:
            result.append(int(item))
    return result or [None]


def _format_cell(value, width):
    if value is None:
        return " " * width
    return f"{value:.1f}".rjust(width)


def post_json_with_retry(url, payload, max_attempts=8, base_sleep_s=0.5):
    for attempt in range(max_attempts):
        try:
            return post_json(url, payload)
        except RuntimeError as exc:
            message = str(exc)
            retryable = any(
                code in message
                for code in (
                    "HTTP error 500",
                    "HTTP error 502",
                    "HTTP error 503",
                    "HTTP error 504",
                    "Loading model",
                )
            )
            if retryable:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(base_sleep_s * (attempt + 1))
                continue
            raise


def run_batch(base_url, prompt, n_predict, concurrency, total_requests, temperature):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_json_with_retry,
                f"{base_url}/completion",
                {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "stream": False,
                },
            )
            for _ in range(total_requests)
        ]
        results = [future.result() for future in as_completed(futures)]
    total_time = time.time() - start_time

    total_tokens = sum(extract_token_count(result) for result in results)
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    return throughput


def _build_server_args(base_args, batch_size, ubatch_size):
    cleaned = []
    skip_next = False
    for arg in base_args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--batch-size", "--ubatch", "-b"}:
            skip_next = True
            continue
        if arg.startswith("--batch-size="):
            continue
        if arg.startswith("--ubatch="):
            continue
        cleaned.append(arg)

    if batch_size is not None:
        cleaned += ["--batch-size", str(batch_size)]
    if ubatch_size is not None:
        cleaned += ["--ubatch", str(ubatch_size)]
    return cleaned


def main():
    prompt = os.environ.get(
        "LLAMA_PROMPT",
        "Share three optimization tips for model serving.",
    )
    temperature = float(os.environ.get("LLAMA_TEMPERATURE", "0.3"))
    instance_count = int(os.environ.get("LLAMA_SERVER_INSTANCES", "2"))
    base_port = int(os.environ.get("LLAMA_SERVER_BASE_PORT", "9000"))
    nginx_port = int(os.environ.get("LLAMA_NGINX_PORT", "8088"))
    ready_timeout_s = int(os.environ.get("LLAMA_READY_TIMEOUT", "180"))
    startup_delay_s = float(os.environ.get("LLAMA_STARTUP_DELAY_S", "0.0"))
    base_args = shlex.split(os.environ.get("LLAMA_SERVER_ARGS", ""))

    max_tokens_list = _parse_int_list(
        os.environ.get("LLAMA_MAX_TOKENS_LIST"),
        "128,256,512,1024",
    )
    batch_list = _parse_optional_int_list(
        os.environ.get("LLAMA_BATCH_LIST"),
        "default",
    )
    ubatch_list = _parse_optional_int_list(
        os.environ.get("LLAMA_UBATCH_LIST"),
        "default",
    )
    concurrency_list = _parse_int_list(
        os.environ.get("LLAMA_CONCURRENCY_LIST"),
        "1,2,4,8,16,32,64,128,256,512,1024",
    )

    total_requests_env = os.environ.get("LLAMA_NUM_REQUESTS")
    requests_multiplier = int(os.environ.get("LLAMA_REQUESTS_MULTIPLIER", "1"))
    cell_pause_s = float(os.environ.get("LLAMA_CELL_PAUSE_S", "0.0"))

    warmup_requests = int(os.environ.get("LLAMA_WARMUP_REQUESTS", "2"))

    if instance_count < 1:
        instance_count = 1
    if requests_multiplier < 1:
        requests_multiplier = 1

    best = {
        "throughput": 0.0,
        "tokens": None,
        "concurrency": None,
        "batch": None,
        "ubatch": None,
    }

    for batch_size in batch_list:
        for ubatch_size in ubatch_list:
            extra_args = _build_server_args(base_args, batch_size, ubatch_size)
            with start_llama_servers(
                instance_count,
                base_port=base_port,
                ready_timeout_s=ready_timeout_s,
                startup_delay_s=startup_delay_s,
                extra_args=extra_args,
            ) as servers:
                upstreams = [(server["host"], server["port"]) for server in servers]
                with start_nginx_round_robin(
                    upstreams,
                    listen_port=nginx_port,
                    listen_host=servers[0]["host"],
                ) as proxy:
                    if warmup_requests > 0:
                        for _ in range(warmup_requests):
                            post_json_with_retry(
                                f"{proxy['base_url']}/completion",
                                {
                                    "prompt": "warmup",
                                    "n_predict": 8,
                                    "temperature": 0.0,
                                    "stream": False,
                                },
                            )

                    batch_label = (
                        "default" if batch_size is None else str(batch_size)
                    )
                    ubatch_label = (
                        "default" if ubatch_size is None else str(ubatch_size)
                    )
                    print(f"\nbatch={batch_label} ubatch={ubatch_label}")
                    col_width = max(7, max(len(str(c)) for c in concurrency_list))
                    header = ["max_tokens \\ conc".rjust(15)]
                    header += [str(c).rjust(col_width) for c in concurrency_list]
                    print(" ".join(header))
                    print("-" * (len(header) * (col_width + 1)))

                    for max_tokens in max_tokens_list:
                        row = [str(max_tokens).rjust(15)]
                        for concurrency in concurrency_list:
                            if total_requests_env:
                                total_requests = int(total_requests_env)
                            else:
                                total_requests = max(
                                    1, concurrency * requests_multiplier
                                )
                            throughput = None
                            try:
                                throughput = run_batch(
                                    proxy["base_url"],
                                    prompt,
                                    max_tokens,
                                    concurrency,
                                    total_requests,
                                    temperature,
                                )
                            except Exception as exc:
                                print(
                                    f"error max_tokens={max_tokens} "
                                    f"concurrency={concurrency}: {exc}",
                                    file=sys.stderr,
                                )
                            if (
                                throughput is not None
                                and throughput > best["throughput"]
                            ):
                                best = {
                                    "throughput": throughput,
                                    "tokens": max_tokens,
                                    "concurrency": concurrency,
                                    "batch": batch_label,
                                    "ubatch": ubatch_label,
                                }
                            row.append(_format_cell(throughput, col_width))
                            if cell_pause_s > 0:
                                time.sleep(cell_pause_s)
                        print(" ".join(row))

    print(
        "best "
        f"max_tokens={best['tokens']} "
        f"concurrency={best['concurrency']} "
        f"batch={best['batch']} "
        f"ubatch={best['ubatch']} "
        f"throughput_tps={best['throughput']:.1f}"
    )


if __name__ == "__main__":
    main()
