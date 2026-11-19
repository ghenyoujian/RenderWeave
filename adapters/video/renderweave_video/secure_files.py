"""Bounded project-path reads and owned-directory writes without symlink traversal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0)
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0)


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def open_absolute_directory(path):
    """Pin every component from / and reject symlinks or replaced directories."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and len(path.parts) <= 128,
            'absolute bounded directory path required')
    fd = os.open('/', DIR_FLAGS)
    try:
        for name in path.parts[1:]:
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode), 'directory component is not a real directory')
            child = os.open(name, DIR_FLAGS, dir_fd=fd)
            try:
                after = os.fstat(child)
                require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                        'directory component changed during open')
            except BaseException:
                os.close(child)
                raise
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def verify_directory(path, fd):
    """Ensure the held directory still occupies the expected absolute path."""
    current = open_absolute_directory(path)
    try:
        left, right = os.fstat(fd), os.fstat(current)
        require((left.st_dev, left.st_ino) == (right.st_dev, right.st_ino), 'owned directory path changed')
        require(os.readlink('/proc/self/fd/' + str(fd)) == str(Path(path)), 'owned directory moved or was unlinked')
    finally:
        os.close(current)


def read_bounded(path, root, maximum):
    """Read an actually opened regular file below root; detect path replacement."""
    path, root = Path(path), Path(root)
    require(path.is_absolute() and root.is_absolute() and '..' not in path.parts and '..' not in root.parts
            and path.is_relative_to(root), 'source is not lexically below the declared root')
    relative = path.relative_to(root)
    require(1 <= len(relative.parts) <= 64 and type(maximum) is int and maximum > 0,
            'source path or byte bound is invalid')
    handles, names = [], []
    file_fd = None
    try:
        handles.append(open_absolute_directory(root))
        for name in relative.parts[:-1]:
            before = os.stat(name, dir_fd=handles[-1], follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode), 'source parent is not a real directory')
            child = os.open(name, DIR_FLAGS, dir_fd=handles[-1])
            try:
                after = os.fstat(child)
                require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                        'source parent changed during open')
            except BaseException:
                os.close(child)
                raise
            names.append(name); handles.append(child)
        name = relative.parts[-1]
        before_path = os.stat(name, dir_fd=handles[-1], follow_symlinks=False)
        require(stat.S_ISREG(before_path.st_mode) and before_path.st_size <= maximum,
                'source is not a bounded regular file')
        file_fd = os.open(name, FILE_FLAGS, dir_fd=handles[-1])
        before_fd = os.fstat(file_fd)
        require(stat.S_ISREG(before_fd.st_mode) and identity(before_path) == identity(before_fd),
                'source changed or is not a regular file during open')
        with os.fdopen(file_fd, 'rb') as stream:
            file_fd = None
            raw = stream.read(before_fd.st_size + 1)
            after_fd = os.fstat(stream.fileno())
        after_path = os.stat(name, dir_fd=handles[-1], follow_symlinks=False)
        require(len(raw) == before_fd.st_size and identity(before_fd) == identity(after_fd) == identity(after_path),
                'source was changed or replaced while reading')
        for index, child_name in enumerate(names):
            observed = os.stat(child_name, dir_fd=handles[index], follow_symlinks=False)
            actual = os.fstat(handles[index + 1])
            require(stat.S_ISDIR(observed.st_mode) and
                    (observed.st_dev, observed.st_ino) == (actual.st_dev, actual.st_ino),
                    'source parent was replaced while reading')
        verify_directory(root, handles[0])
        return raw
    except OSError as error:
        raise ValueError('source path unavailable, replaced or contains a symlink') from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for fd in reversed(handles):
            os.close(fd)


def json_snapshot(path, root, maximum=16 * 1024 * 1024):
    """Return one bounded verified byte snapshot, its object and its SHA-256."""
    raw = read_bounded(path, root, maximum)
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key in source manifest')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(value, dict), 'source manifest must be an object')
    return raw, value, hashlib.sha256(raw).hexdigest()
