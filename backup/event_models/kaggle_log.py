"""Print a Kaggle kernel's log WITHOUT downloading any output files.

`kaggle kernels output` pulls every output file to disk; this reads only the
`log` field of the same API response (the pattern used for the other projects'
long runs).  Usage:  python kaggle_log.py owner/slug [tail_lines]
"""
import json
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def fetch_log(kernel: str) -> str:
    owner, slug = kernel.split("/")
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    raw = response.log or ""
    try:  # Kaggle serialises the log as a JSON list of {stream_name, time, data}
        return "".join(item.get("data", "") for item in json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        return raw


if __name__ == "__main__":
    text = fetch_log(sys.argv[1])
    lines = text.splitlines()
    keep = int(sys.argv[2]) if len(sys.argv) > 2 else len(lines)
    print("\n".join(lines[-keep:]))
