"""Shared OpenAI Responses API access for the RIFAI CLI and website."""

import os

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        OpenAI,
        RateLimitError,
    )
except ImportError:  # Keep local/offline parts usable without the SDK installed.
    APIConnectionError = APIStatusError = APITimeoutError = None
    AuthenticationError = RateLimitError = None
    OpenAI = None


MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna"
API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=API_KEY, timeout=30.0, max_retries=1) if API_KEY and OpenAI else None


class RifaiAPIError(Exception):
    """An API problem with a safe, user-facing explanation."""

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def ask_rifai(message, history=None, instructions=None, max_output_tokens=800):
    """Send a Responses API request and return plain text, without leaking secrets."""
    if not API_KEY:
        raise RifaiAPIError(
            "API AI belum dikonfigurasi. Atur OPENAI_API_KEY di environment lalu mulai ulang.",
            status_code=503,
        )
    if OpenAI is None:
        raise RifaiAPIError(
            "Pustaka OpenAI belum tersedia. Pasang dependensi proyek sebelum memakai AI API.",
            status_code=503,
        )
    if client is None:
        raise RifaiAPIError("Klien AI belum siap. Periksa konfigurasi server lalu mulai ulang.")

    request_input = list(history or [])
    request_input.append({"role": "user", "content": message})
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=instructions or "Jawab dengan ramah dan jelas dalam bahasa pengguna.",
            input=request_input,
            max_output_tokens=max_output_tokens,
        )
    except AuthenticationError:
        raise RifaiAPIError(
            "Autentikasi OpenAI gagal. Periksa OPENAI_API_KEY yang aktif dan mulai ulang."
        ) from None
    except RateLimitError:
        raise RifaiAPIError(
            "Batas permintaan atau kuota OpenAI tercapai. Periksa kuota/billing lalu coba lagi.",
            status_code=429,
        ) from None
    except APITimeoutError:
        raise RifaiAPIError(
            "Permintaan ke OpenAI melewati batas waktu. Periksa koneksi lalu coba lagi.",
            status_code=504,
        ) from None
    except APIConnectionError:
        raise RifaiAPIError(
            "Tidak dapat terhubung ke OpenAI. Periksa koneksi internet lalu coba lagi."
        ) from None
    except APIStatusError as error:
        status = error.status_code
        if status == 404:
            message = "Model OpenAI tidak tersedia untuk akun ini. Periksa OPENAI_MODEL."
        elif status == 400:
            message = "Permintaan atau model tidak diterima. Periksa OPENAI_MODEL dan konfigurasi."
        elif status >= 500:
            message = "Layanan OpenAI sedang mengalami gangguan. Coba lagi beberapa saat."
        else:
            message = f"Permintaan OpenAI ditolak (HTTP {status}). Periksa konfigurasi akun."
        raise RifaiAPIError(message) from None
    except Exception as error:
        # Never include exception text: SDK exceptions may contain sensitive request details.
        raise RifaiAPIError(
            "Terjadi kesalahan saat meminta jawaban AI. Coba lagi nanti."
        ) from None

    reply = getattr(response, "output_text", "")
    if not isinstance(reply, str) or not reply.strip():
        raise RifaiAPIError("AI belum menghasilkan jawaban. Silakan coba kembali.")
    return reply.strip()
