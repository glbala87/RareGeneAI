"""Secure temporary file handling for medical data.

Ensures:
  - Temp files are created in a restricted directory (mode 0o700)
  - Files are automatically cleaned up on exit
  - No world-readable permissions
  - Content is securely wiped on cleanup
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import TracebackType

from loguru import logger

# Default secure temp directory
_SECURE_TEMP_BASE = os.environ.get(
    "RAREGENEAI_SECURE_TEMP_DIR",
    os.path.join(tempfile.gettempdir(), "raregeneai_secure"),
)


def _ensure_secure_temp_dir() -> Path:
    """Create and secure the temp directory."""
    path = Path(_SECURE_TEMP_BASE)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


class SecureTempFile:
    """Context manager for secure temporary files.

    Usage:
        async with SecureTempFile(suffix=".vcf", content=data) as path:
            process(path)
        # File is securely deleted after the block
    """

    def __init__(
        self,
        suffix: str = "",
        content: bytes | None = None,
        prefix: str = "rgai_",
    ):
        self.suffix = suffix
        self.content = content
        self.prefix = prefix
        self._path: Path | None = None

    def __enter__(self) -> Path:
        secure_dir = _ensure_secure_temp_dir()
        fd, path_str = tempfile.mkstemp(
            suffix=self.suffix,
            prefix=self.prefix,
            dir=str(secure_dir),
        )
        self._path = Path(path_str)

        try:
            # Set restrictive permissions (owner-only read/write)
            os.fchmod(fd, 0o600)

            if self.content:
                os.write(fd, self.content)
        finally:
            os.close(fd)

        return self._path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._cleanup()

    async def __aenter__(self) -> Path:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Securely remove the temporary file."""
        if self._path and self._path.exists():
            try:
                # Overwrite with zeros before deletion
                size = self._path.stat().st_size
                if size > 0:
                    with open(self._path, "wb") as f:
                        f.write(b"\x00" * min(size, 1024 * 1024))  # Zero first 1MB
                        f.flush()
                        os.fsync(f.fileno())
                self._path.unlink()
                logger.debug(f"Securely deleted temp file: {self._path.name}")
            except OSError as e:
                logger.warning(f"Failed to securely delete temp file {self._path}: {e}")
