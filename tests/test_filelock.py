from __future__ import annotations

import errno
import inspect
import logging
import os
import subprocess  # noqa: S404  # only fixed-argument, no-shell icacls runs below
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from errno import EIO, ENOSYS, EWOULDBLOCK
from inspect import getframeinfo, stack
from pathlib import Path, PurePath
from stat import S_IMODE, S_IWGRP, S_IWOTH, S_IWUSR, filemode
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Iterator, Tuple, Type, Union
from uuid import uuid4

import pytest
from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout, UnixFileLock, WindowsFileLock

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    ("lock_type", "path_type"),
    [
        (FileLock, str),
        (FileLock, PurePath),
        (FileLock, Path),
        (SoftFileLock, str),
        (SoftFileLock, PurePath),
        (SoftFileLock, Path),
    ],
)
def test_simple(
    lock_type: type[BaseFileLock],
    path_type: type[str] | type[Path],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    # test lock creation by passing a `str`
    lock_path = tmp_path / "a"
    lock = lock_type(path_type(lock_path))
    with lock as locked:
        assert lock.is_locked
        assert lock is locked
    assert not lock.is_locked

    assert caplog.messages == [
        f"Attempting to acquire lock {id(lock)} on {lock_path}",
        f"Lock {id(lock)} acquired on {lock_path}",
        f"Attempting to release lock {id(lock)} on {lock_path}",
        f"Lock {id(lock)} released on {lock_path}",
    ]
    assert [r.levelno for r in caplog.records] == [logging.DEBUG, logging.DEBUG, logging.DEBUG, logging.DEBUG]
    assert [r.name for r in caplog.records] == ["filelock", "filelock", "filelock", "filelock"]
    assert logging.getLogger("filelock").level == logging.NOTSET


@contextmanager
def make_ro(path: Path) -> Iterator[None]:
    write = S_IWUSR | S_IWGRP | S_IWOTH
    path.chmod(path.stat().st_mode & ~write)
    yield
    path.chmod(path.stat().st_mode | write)


@pytest.fixture()
def tmp_path_ro(tmp_path: Path) -> Iterator[Path]:
    with make_ro(tmp_path):
        yield tmp_path


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not have read only folders")
@pytest.mark.skipif(
    sys.platform != "win32" and os.geteuid() == 0,
    reason="Cannot make a read only file (that the current user: root can't read)",
)
def test_ro_folder(lock_type: type[BaseFileLock], tmp_path_ro: Path) -> None:
    lock = lock_type(str(tmp_path_ro / "a"))
    with pytest.raises(PermissionError, match="Permission denied"):
        lock.acquire()


@pytest.fixture()
def tmp_file_ro(tmp_path: Path) -> Iterator[Path]:
    filename = tmp_path / "a"
    filename.write_text("")
    with make_ro(filename):
        yield filename


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
@pytest.mark.skipif(
    sys.platform != "win32" and os.geteuid() == 0,
    reason="Cannot make a read only file (that the current user: root can't read)",
)
def test_ro_file(lock_type: type[BaseFileLock], tmp_file_ro: Path) -> None:
    lock = lock_type(str(tmp_file_ro))
    with pytest.raises(PermissionError, match="Permission denied"):
        lock.acquire()


WindowsOnly = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
@pytest.mark.parametrize(
    ("expected_error", "match", "bad_lock_file"),
    [
        # WindowsFileLock raises the real Win32 error now, so accept its wording alongside os.open's.
        pytest.param(
            FileNotFoundError,
            "No such file or directory:|cannot find the (path|file)",
            "a/b",
            id="non_existent_directory",
        ),
        pytest.param(
            FileNotFoundError, "No such file or directory:|cannot find the (path|file)", "", id="blank_filename"
        ),
        pytest.param(ValueError, "embedded null (byte|character)", "\0", id="null_byte"),
        # Should be PermissionError on Windows
        pytest.param(PermissionError, "Permission denied:", ".", id="current_directory")
        if sys.platform == "win32"
        else (
            # Should be IsADirectoryError on MacOS and Linux
            pytest.param(IsADirectoryError, "Is a directory", ".", id="current_directory")
            if sys.platform in ["darwin", "linux"]
            else
            # Should be some type of OSError at least on other operating systems
            pytest.param(OSError, None, ".", id="current_directory")
        ),
    ]
    + [
        pytest.param(OSError, "Invalid argument|syntax is incorrect", i, id=f"invalid_{i}", marks=WindowsOnly)
        for i in '<>:"|?*\a'
    ]
    + [pytest.param(PermissionError, "Permission denied:", i, id=f"permission_{i}", marks=WindowsOnly) for i in "/\\"],
)
@pytest.mark.timeout(5)  # timeout in case of infinite loop
def test_bad_lock_file(
    lock_type: type[BaseFileLock],
    expected_error: type[Exception],
    match: str,
    bad_lock_file: str,
) -> None:
    lock = lock_type(bad_lock_file)

    with pytest.raises(expected_error, match=match):
        lock.acquire()


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_nested_context_manager(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # lock is not released before the most outer with statement that locked the lock, is left
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    with lock as lock_1:
        assert lock.is_locked
        assert lock is lock_1

        with lock as lock_2:
            assert lock.is_locked
            assert lock is lock_2

            with lock as lock_3:
                assert lock.is_locked
                assert lock is lock_3

            assert lock.is_locked
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_nested_acquire(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # lock is not released before the most outer with statement that locked the lock, is left
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    with lock.acquire() as lock_1:
        assert lock.is_locked
        assert lock is lock_1

        with lock.acquire() as lock_2:
            assert lock.is_locked
            assert lock is lock_2

            with lock.acquire() as lock_3:
                assert lock.is_locked
                assert lock is lock_3

            assert lock.is_locked
        assert lock.is_locked
    assert not lock.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_nested_forced_release(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # acquires the lock using a with-statement and releases the lock before leaving the with-statement
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    with lock:
        assert lock.is_locked

        lock.acquire()
        assert lock.is_locked

        lock.release(force=True)
        assert not lock.is_locked
    assert not lock.is_locked


_ExcInfoType = Union[Tuple[Type[BaseException], BaseException, TracebackType], Tuple[None, None, None]]


class ExThread(threading.Thread):
    def __init__(self, target: Callable[[], None], name: str) -> None:
        super().__init__(target=target, name=name)
        self.ex: _ExcInfoType | None = None

    def run(self) -> None:
        try:
            super().run()
        except Exception:  # noqa: BLE001 # pragma: no cover
            self.ex = sys.exc_info()  # pragma: no cover

    def join(self, timeout: float | None = None) -> None:
        super().join(timeout=timeout)
        if self.ex is not None:
            raise RuntimeError from self.ex[1]  # pragma: no cover


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_threaded_shared_lock_obj(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # Runs 100 threads, which need the filelock. The lock must be acquired if at least one thread required it and
    # released, as soon as all threads stopped.
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    def thread_work() -> None:
        for _ in range(100):
            with lock:
                assert lock.is_locked

    threads = [ExThread(target=thread_work, name=f"t{i}") for i in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not lock.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
@pytest.mark.skipif(hasattr(sys, "pypy_version_info") and sys.platform == "win32", reason="deadlocks randomly")
def test_threaded_lock_different_lock_obj(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # Runs multiple threads, which acquire the same lock file with a different FileLock object. When thread group 1
    # acquired the lock, thread group 2 must not hold their lock.

    def t_1() -> None:
        for _ in range(1000):
            with lock_1:
                assert lock_1.is_locked
                assert not lock_2.is_locked

    def t_2() -> None:
        for _ in range(1000):
            with lock_2:
                assert not lock_1.is_locked
                assert lock_2.is_locked

    lock_path = tmp_path / "a"
    lock_1, lock_2 = lock_type(str(lock_path)), lock_type(str(lock_path))
    threads = [(ExThread(t_1, f"t1_{i}"), ExThread(t_2, f"t2_{i}")) for i in range(10)]

    for thread_1, thread_2 in threads:
        thread_1.start()
        thread_2.start()
    for thread_1, thread_2 in threads:
        thread_1.join()
        thread_2.join()

    assert not lock_1.is_locked
    assert not lock_2.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_timeout(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # raises Timeout error when the lock cannot be acquired
    lock_path = tmp_path / "a"
    lock_1, lock_2 = lock_type(str(lock_path)), lock_type(str(lock_path))

    # acquire lock 1
    lock_1.acquire()
    assert lock_1.is_locked
    assert not lock_2.is_locked

    # try to acquire lock 2
    with pytest.raises(Timeout, match="The file lock '.*' could not be acquired."):
        lock_2.acquire(timeout=0.1)
    assert not lock_2.is_locked
    assert lock_1.is_locked

    # release lock 1
    lock_1.release()
    assert not lock_1.is_locked
    assert not lock_2.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_non_blocking(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # raises Timeout error when the lock cannot be acquired
    lock_path = tmp_path / "a"
    lock_1, lock_2 = lock_type(str(lock_path)), lock_type(str(lock_path))

    # acquire lock 1
    lock_1.acquire()
    assert lock_1.is_locked
    assert not lock_2.is_locked

    # try to acquire lock 2
    with pytest.raises(Timeout, match="The file lock '.*' could not be acquired."):
        lock_2.acquire(blocking=False)
    assert not lock_2.is_locked
    assert lock_1.is_locked

    # release lock 1
    lock_1.release()
    assert not lock_1.is_locked
    assert not lock_2.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_default_timeout(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # test if the default timeout parameter works
    lock_path = tmp_path / "a"
    lock_1, lock_2 = lock_type(str(lock_path)), lock_type(str(lock_path), timeout=0.1)
    assert lock_2.timeout == 0.1

    # acquire lock 1
    lock_1.acquire()
    assert lock_1.is_locked
    assert not lock_2.is_locked

    # try to acquire lock 2
    with pytest.raises(Timeout, match="The file lock '.*' could not be acquired."):
        lock_2.acquire()
    assert not lock_2.is_locked
    assert lock_1.is_locked

    lock_2.timeout = 0
    assert lock_2.timeout == 0

    with pytest.raises(Timeout, match="The file lock '.*' could not be acquired."):
        lock_2.acquire()
    assert not lock_2.is_locked
    assert lock_1.is_locked

    # release lock 1
    lock_1.release()
    assert not lock_1.is_locked
    assert not lock_2.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_context_release_on_exc(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # lock is released when an exception is thrown in a with-statement
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    try:
        with lock as lock_1:
            assert lock is lock_1
            assert lock.is_locked
            raise ValueError  # noqa: TRY301
    except ValueError:
        assert not lock.is_locked


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_acquire_release_on_exc(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # lock is released when an exception is thrown in a acquire statement
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    try:
        with lock.acquire() as lock_1:
            assert lock is lock_1
            assert lock.is_locked
            raise ValueError  # noqa: TRY301
    except ValueError:
        assert not lock.is_locked


@pytest.mark.skipif(hasattr(sys, "pypy_version_info"), reason="del() does not trigger GC in PyPy")
@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_del(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    # lock is released when the object is deleted
    lock_path = tmp_path / "a"
    lock_1, lock_2 = lock_type(str(lock_path)), lock_type(str(lock_path))

    # acquire lock 1
    lock_1.acquire()
    assert lock_1.is_locked
    assert not lock_2.is_locked

    # try to acquire lock 2
    with pytest.raises(Timeout, match="The file lock '.*' could not be acquired."):
        lock_2.acquire(timeout=0.1)

    # delete lock 1 and try to acquire lock 2 again
    del lock_1

    lock_2.acquire()
    assert lock_2.is_locked

    lock_2.release()


def test_cleanup_soft_lock(tmp_path: Path) -> None:
    # tests if the lock file is removed after use
    lock_path = tmp_path / "a"

    with SoftFileLock(lock_path):
        assert lock_path.exists()
    assert not lock_path.exists()


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_poll_intervall_deprecated(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    with pytest.deprecated_call(match="use poll_interval instead of poll_intervall") as checker:
        lock.acquire(poll_intervall=0.05)  # the deprecation warning will be captured by the checker
        frame_info = getframeinfo(stack()[0][0])  # get frame info of current file and lineno (+1 than the above lineno)
        for warning in checker:
            if warning.filename == frame_info.filename and warning.lineno + 1 == frame_info.lineno:  # pragma: no cover
                break
        else:  # pragma: no cover
            pytest.fail("No warnings of stacklevel=2 matching.")


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_context_decorator(lock_type: type[BaseFileLock], tmp_path: Path) -> None:
    lock_path = tmp_path / "a"
    lock = lock_type(str(lock_path))

    @lock
    def decorated_method() -> None:
        assert lock.is_locked

    assert not lock.is_locked
    decorated_method()
    assert not lock.is_locked


def test_lock_mode(tmp_path: Path) -> None:
    # test file lock permissions are independent of umask
    lock_path = tmp_path / "a.lock"
    lock = FileLock(str(lock_path), mode=0o666)

    # set umask so permissions can be anticipated
    initial_umask = os.umask(0o022)
    try:
        lock.acquire()
        assert lock.is_locked

        mode = filemode(lock_path.stat().st_mode)
        assert mode == "-rw-rw-rw-"
    finally:
        os.umask(initial_umask)

    lock.release()


def test_lock_mode_soft(tmp_path: Path) -> None:
    # test soft lock permissions are dependent of umask
    lock_path = tmp_path / "a.lock"
    lock = SoftFileLock(str(lock_path), mode=0o666)

    # set umask so permissions can be anticipated
    initial_umask = os.umask(0o022)
    try:
        lock.acquire()
        assert lock.is_locked

        mode = filemode(lock_path.stat().st_mode)
        if sys.platform == "win32":
            assert mode == "-rw-rw-rw-"
        else:
            assert mode == "-rw-r--r--"
    finally:
        os.umask(initial_umask)

    lock.release()


def test_umask(tmp_path: Path) -> None:
    lock_path = tmp_path / "a.lock"
    lock = FileLock(str(lock_path), mode=0o666)

    initial_umask = os.umask(0)
    os.umask(initial_umask)

    lock.acquire()
    assert lock.is_locked

    current_umask = os.umask(0)
    os.umask(current_umask)
    assert initial_umask == current_umask

    lock.release()


def test_umask_soft(tmp_path: Path) -> None:
    lock_path = tmp_path / "a.lock"
    lock = SoftFileLock(str(lock_path), mode=0o666)

    initial_umask = os.umask(0)
    os.umask(initial_umask)

    lock.acquire()
    assert lock.is_locked

    current_umask = os.umask(0)
    os.umask(current_umask)
    assert initial_umask == current_umask

    lock.release()


def test_wrong_platform(tmp_path: Path) -> None:
    assert not inspect.isabstract(UnixFileLock)
    assert not inspect.isabstract(WindowsFileLock)
    assert inspect.isabstract(BaseFileLock)

    lock_type = UnixFileLock if sys.platform == "win32" else WindowsFileLock
    lock = lock_type(tmp_path / "lockfile")

    with pytest.raises(NotImplementedError):
        lock.acquire()
    with pytest.raises(NotImplementedError):
        lock._release()  # noqa: SLF001


@pytest.mark.skipif(sys.platform == "win32", reason="flock not run on windows")
def test_flock_not_implemented_unix(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("fcntl.flock", side_effect=OSError(ENOSYS, "mock error"))
    with pytest.raises(NotImplementedError), FileLock(tmp_path / "a.lock"):
        pass


def test_soft_errors(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("os.open", side_effect=OSError(ENOSYS, "mock error"))
    with pytest.raises(OSError, match="mock error"):
        SoftFileLock(tmp_path / "a.lock").acquire()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="needs O_NOFOLLOW")
def test_soft_lock_opens_with_o_nofollow(tmp_path: Path, mocker: MockerFixture) -> None:
    spy = mocker.spy(os, "open")
    lock = SoftFileLock(str(tmp_path / "a.lock"))
    with lock:
        pass
    assert spy.call_count >= 1
    for call in spy.call_args_list:
        assert call[0][1] & os.O_NOFOLLOW


def _check_file_read_write(txt_file: Path) -> None:
    for _ in range(3):
        uuid = str(uuid4())
        txt_file.write_text(uuid)
        assert txt_file.read_text() == uuid


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_thrashing_with_thread_pool_passing_lock_to_threads(tmp_path: Path, lock_type: type[BaseFileLock]) -> None:
    def mess_with_file(lock_: BaseFileLock) -> None:
        with lock_:
            _check_file_read_write(txt_file)

    lock_file, txt_file = tmp_path / "test.txt.lock", tmp_path / "test.txt"
    lock = lock_type(lock_file)
    results = []
    with ThreadPoolExecutor() as executor:
        for _ in range(100):
            results.append(executor.submit(mess_with_file, lock))

    assert all(r.result() is None for r in results)


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_thrashing_with_thread_pool_global_lock(tmp_path: Path, lock_type: type[BaseFileLock]) -> None:
    def mess_with_file() -> None:
        with lock:
            _check_file_read_write(txt_file)

    lock_file, txt_file = tmp_path / "test.txt.lock", tmp_path / "test.txt"
    lock = lock_type(lock_file)
    results = []
    with ThreadPoolExecutor() as executor:
        for _ in range(100):
            results.append(executor.submit(mess_with_file))

    assert all(r.result() is None for r in results)


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_thrashing_with_thread_pool_lock_recreated_in_each_thread(
    tmp_path: Path,
    lock_type: type[BaseFileLock],
) -> None:
    def mess_with_file() -> None:
        with lock_type(lock_file):
            _check_file_read_write(txt_file)

    lock_file, txt_file = tmp_path / "test.txt.lock", tmp_path / "test.txt"
    results = []
    with ThreadPoolExecutor() as executor:
        for _ in range(100):
            results.append(executor.submit(mess_with_file))

    assert all(r.result() is None for r in results)


@pytest.mark.parametrize("lock_type", [FileLock, SoftFileLock])
def test_lock_can_be_non_thread_local(
    tmp_path: Path,
    lock_type: type[BaseFileLock],
) -> None:
    lock = lock_type(tmp_path / "test.lock", thread_local=False)

    for _ in range(2):
        thread = threading.Thread(target=lock.acquire, kwargs={"timeout": 2})
        thread.start()
        thread.join()

    assert lock.lock_counter == 2

    lock.release(force=True)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except NotImplementedError:  # PyPy before 3.9 has no os.symlink on Windows, so call the Win32 API directly
        if sys.platform == "win32":  # only the Windows-only tests get here; the guard narrows ctypes.WinDLL for mypy
            import ctypes

            create_symbolic_link = ctypes.WinDLL("kernel32", use_last_error=True).CreateSymbolicLinkW
            create_symbolic_link.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
            create_symbolic_link.restype = ctypes.c_ubyte
            # 0x2 is SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE, honoured when Developer Mode is on
            if create_symbolic_link(str(link), str(target), 0x2):
                return
        pytest.skip("cannot create symlinks (needs Developer Mode or administrator)")
    except OSError:
        pytest.skip("cannot create symlinks (needs Developer Mode or administrator)")


@WindowsOnly
def test_windows_reparse_point_lock_file_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("sensitive", encoding="utf-8")
    link = tmp_path / "resource.lock"
    _symlink_or_skip(link, target)

    with pytest.raises(OSError, match="reparse point"):
        FileLock(link).acquire()

    assert target.read_text(encoding="utf-8") == "sensitive"


@WindowsOnly
def test_windows_dangling_reparse_point_lock_file_rejected(tmp_path: Path) -> None:
    # a dangling symlink planted at the lock path must not make the lock create the file it points to
    target = tmp_path / "target.txt"
    link = tmp_path / "resource.lock"
    _symlink_or_skip(link, target)

    with pytest.raises(OSError, match="reparse point"):
        FileLock(link, timeout=0).acquire()

    assert not target.exists()


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "O_NOFOLLOW"), reason="needs POSIX O_NOFOLLOW")
@pytest.mark.parametrize("target_exists", [True, False], ids=["existing_target", "dangling_target"])
def test_unix_symlink_lock_file_not_followed(tmp_path: Path, target_exists: bool) -> None:  # noqa: FBT001
    # a symlink planted at the lock path must not make the lock truncate or create the file it points to
    target = tmp_path / "target.txt"
    if target_exists:
        target.write_text("sensitive", encoding="utf-8")
    link = tmp_path / "resource.lock"
    link.symlink_to(target)

    lock = UnixFileLock(link, timeout=0)
    with pytest.raises(OSError) as exc_info:  # noqa: PT011
        lock.acquire()

    assert not isinstance(exc_info.value, Timeout)
    assert exc_info.value.errno in (errno.ELOOP, errno.EMLINK)  # EMLINK is what FreeBSD reports for O_NOFOLLOW
    assert not lock.is_locked
    if target_exists:
        assert target.read_text(encoding="utf-8") == "sensitive"
    else:
        assert not target.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="flock not run on windows")
@pytest.mark.parametrize("mode", [pytest.param(None, id="default-mode"), pytest.param(0o666, id="explicit-mode")])
def test_winner_leaves_existing_file_as_found(tmp_path: Path, mode: int | None) -> None:
    journal = tmp_path / "journal.log"
    journal.write_text("data", encoding="utf-8")
    journal.chmod(0o600)
    # A regular file with one name is indistinguishable from a lock file an earlier holder left.
    lock_path = tmp_path / "resource.lock"
    journal.rename(lock_path)

    with FileLock(lock_path, timeout=0) if mode is None else FileLock(lock_path, timeout=0, mode=mode):
        assert (lock_path.read_text(encoding="utf-8"), S_IMODE(lock_path.stat().st_mode)) == ("data", 0o600)


@pytest.mark.skipif(sys.platform == "win32", reason="flock not run on windows")
def test_winner_leaves_hard_linked_file_as_found(tmp_path: Path) -> None:
    journal = tmp_path / "journal.log"
    journal.write_text("data", encoding="utf-8")
    journal.chmod(0o600)
    # A hard link to another file at the lock path must not let the lock empty that file or widen its mode.
    lock_path = tmp_path / "resource.lock"
    os.link(str(journal), str(lock_path))

    with FileLock(lock_path, timeout=0, mode=0o666):
        pass

    assert (journal.read_text(encoding="utf-8"), S_IMODE(journal.stat().st_mode)) == ("data", 0o600)


@pytest.mark.skipif(sys.platform == "win32", reason="flock not run on windows")
def test_creator_that_loses_the_flock_applies_mode(tmp_path: Path, mocker: MockerFixture) -> None:
    lock_path = tmp_path / "resource.lock"
    # A peer may win the flock first, and only the creator may set the mode.
    mocker.patch("filelock._unix.fcntl.flock", side_effect=OSError(EWOULDBLOCK, "Resource temporarily unavailable"))

    initial_umask = os.umask(0o077)
    try:
        with pytest.raises(Timeout):
            FileLock(lock_path, timeout=0, mode=0o666).acquire()
    finally:
        os.umask(initial_umask)

    assert S_IMODE(lock_path.stat().st_mode) == 0o666


# The upstream test also has an "fstat" case: there the winner fstats the locked descriptor to check its link count,
# a step this version's acquire does not have, so only the fchmod a creator runs after the open can fail here.
@pytest.mark.skipif(sys.platform == "win32", reason="flock not run on windows")
@pytest.mark.parametrize("failing", [pytest.param("fchmod", id="fchmod")])
def test_post_open_failure_closes_fd(tmp_path: Path, mocker: MockerFixture, failing: str) -> None:
    mocker.patch(f"os.{failing}", side_effect=OSError(EIO, "Input/output error"))
    open_spy = mocker.spy(os, "open")
    close_spy = mocker.spy(os, "close")

    with pytest.raises(OSError, match="Input/output error"):
        FileLock(tmp_path / "resource.lock", timeout=0, mode=0o644).acquire()

    assert any(call[0] and call[0][0] == open_spy.spy_return for call in close_spy.call_args_list)


@pytest.mark.skipif(sys.platform == "win32", reason="flock not run on windows")
def test_fchmod_enosys_is_not_a_missing_flock(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("os.fchmod", side_effect=OSError(ENOSYS, "Function not implemented"))
    lock = UnixFileLock(tmp_path / "test.lock", mode=0o644)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(OSError, match="Function not implemented"):
            lock.acquire()

    assert not recorded
    assert not lock.is_locked


@pytest.mark.parametrize("mode", [0o644, 0o666, 0o600])
def test_existing_lock_file_keeps_its_contents(tmp_path: Path, mode: int) -> None:
    lock_path = tmp_path / "resource.lock"
    lock_path.write_bytes(b"data")

    with FileLock(str(lock_path), timeout=0, mode=mode):
        # The size rather than the content: Windows refuses a read of the byte the lock holds.
        assert lock_path.stat().st_size == len(b"data")


def test_owner_only_mode_keeps_mutual_exclusion(tmp_path: Path) -> None:
    lock_path = tmp_path / "a"
    lock_1, lock_2 = FileLock(str(lock_path), mode=0o600), FileLock(str(lock_path), mode=0o600)

    with lock_1:
        with pytest.raises(Timeout):
            lock_2.acquire(timeout=0.1)
        assert not lock_2.is_locked
    with lock_2:
        assert lock_2.is_locked


def test_owner_only_mode_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileLock(str(tmp_path / "missing" / "a"), timeout=0, mode=0o600).acquire()


def _icacls(path: Path, *args: str) -> bytes:  # pragma: win32 cover
    icacls = Path(os.environ["SYSTEMROOT"], "System32", "icacls.exe")
    return subprocess.run([str(icacls), str(path), *args], capture_output=True, check=True).stdout  # noqa: S603


@pytest.fixture()
def inheriting_directory(tmp_path: Path) -> Path:  # pragma: win32 cover
    # A directory whose ACL has no inheritable entry hands a new file the token's default DACL, which icacls cannot
    # tell from an explicit one, so grant Users an inheritable entry for the file to show as (I).
    directory = tmp_path / "inheriting"
    directory.mkdir()
    _icacls(directory, "/grant", "*S-1-5-32-545:(OI)(CI)(RX)")
    return directory


@WindowsOnly
@pytest.mark.parametrize(
    ("mode", "inherits"),
    [
        pytest.param(None, True, id="unset"),
        pytest.param(0o644, True, id="others-read"),
        pytest.param(0o666, True, id="others-write"),
        pytest.param(0o600, False, id="owner-only"),
    ],
)
def test_windows_owner_only_mode_keeps_out_the_directory_acl(
    inheriting_directory: Path,
    mode: int | None,
    *,
    inherits: bool,
) -> None:
    lock_path = inheriting_directory / "a"
    with FileLock(str(lock_path)) if mode is None else FileLock(str(lock_path), mode=mode):
        acl = _icacls(lock_path)
    assert (b"(I)" in acl) is inherits  # icacls marks inherited entries with (I) in every display language


@WindowsOnly
def test_windows_owner_only_mode_leaves_an_existing_file_acl(inheriting_directory: Path) -> None:
    # As on Unix, the mode only applies to a file the lock creates; opening an existing one keeps its ACL.
    lock_path = inheriting_directory / "a"
    lock_path.write_bytes(b"data")

    with FileLock(str(lock_path), mode=0o600):
        acl = _icacls(lock_path)
        assert lock_path.stat().st_size == len(b"data")
    assert b"(I)" in acl


@WindowsOnly
def test_windows_owner_only_mode_rejects_reparse_point(tmp_path: Path) -> None:
    # The owner-only open goes through the same reparse check, bound to the opened handle, as any other mode.
    target = tmp_path / "target.txt"
    target.write_text("sensitive", encoding="utf-8")
    link = tmp_path / "resource.lock"
    _symlink_or_skip(link, target)

    with pytest.raises(OSError, match="reparse point"):
        FileLock(link, timeout=0, mode=0o600).acquire()

    assert target.read_text(encoding="utf-8") == "sensitive"
